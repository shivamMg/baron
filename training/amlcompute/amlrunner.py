#!/usr/bin/env python3
"""Manage durable Blob storage services on an Azure ML compute node."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import paramiko
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
REMOTE_BOOTSTRAP = "/tmp/baron-bootstrap.sh"
LOCAL_BOOTSTRAP = BASE_DIR / "remote" / "bootstrap.sh"
STATE_DIR = BASE_DIR / ".state"
KNOWN_HOSTS = STATE_DIR / "known_hosts"

AML_API_VERSION = "2024-10-01"
STORAGE_API_VERSION = "2023-05-01"
IDENTITY_API_VERSION = "2023-01-31"


class AutomationError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AutomationError(f"Missing required setting {name} in {ENV_FILE}")
    return value


def load_settings() -> dict[str, str]:
    load_dotenv(ENV_FILE)
    return {
        "aml_compute_id": required_env("AML_COMPUTE_RESOURCE_ID").rstrip("/"),
        "storage_account_id": required_env("STORAGE_ACCOUNT_RESOURCE_ID").rstrip("/"),
        "identity_id": required_env("MANAGED_IDENTITY_RESOURCE_ID").rstrip("/"),
        "storage_container": required_env("STORAGE_CONTAINER"),
        "ssh_user": required_env("AML_SSH_USER"),
        "ssh_password": required_env("AML_SSH_PASSWORD"),
    }


def resource_name(resource_id: str) -> str:
    parts = [part for part in resource_id.split("/") if part]
    if not parts:
        raise AutomationError(f"Invalid Azure resource ID: {resource_id}")
    return parts[-1]


def az_json(*args: str) -> Any:
    executable = shutil.which("az.cmd" if os.name == "nt" else "az")
    if executable is None:
        raise AutomationError("Azure CLI was not found on PATH")
    command = [executable, *args, "--only-show-errors", "--output", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise AutomationError(f"Azure CLI failed: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AutomationError("Azure CLI returned invalid JSON") from exc


def arm_url(resource_id: str, api_version: str) -> str:
    return f"https://management.azure.com{resource_id}?api-version={api_version}"


def ensure_storage_container(settings: dict[str, str]) -> None:
    container_id = (
        f"{settings['storage_account_id']}/blobServices/default/containers/"
        f"{settings['storage_container']}"
    )
    az_json(
        "rest",
        "--method",
        "put",
        "--url",
        arm_url(container_id, STORAGE_API_VERSION),
        "--body",
        '{"properties":{}}',
    )


def get_identity_client_id(settings: dict[str, str]) -> str:
    identity = az_json(
        "rest",
        "--method",
        "get",
        "--url",
        arm_url(settings["identity_id"], IDENTITY_API_VERSION),
    )
    return identity["properties"]["clientId"]


def get_storage_account_key(settings: dict[str, str]) -> str:
    keys = az_json(
        "rest",
        "--method",
        "post",
        "--url",
        arm_url(f"{settings['storage_account_id']}/listKeys", STORAGE_API_VERSION),
    ).get("keys", [])
    if not keys or not keys[0].get("value"):
        raise AutomationError("Storage account listKeys returned no usable key")
    return keys[0]["value"]


def discover_node(settings: dict[str, str]) -> dict[str, Any]:
    response = az_json(
        "rest",
        "--method",
        "post",
        "--url",
        arm_url(f"{settings['aml_compute_id']}/listNodes", AML_API_VERSION),
    )
    nodes = response.get("nodes", response.get("value", response))
    if isinstance(nodes, dict):
        nodes = [nodes]
    usable = [
        node for node in nodes if node.get("publicIpAddress") and node.get("port")
    ]
    if not usable:
        raise AutomationError("AML returned no running node with a public SSH endpoint")
    usable.sort(key=lambda node: node.get("nodeState", "") != "idle")
    node = usable[0]
    return {
        "node_id": node.get("nodeId", "unknown-node"),
        "state": node.get("nodeState", "unknown"),
        "host": node["publicIpAddress"],
        "port": int(node["port"]),
        "ssh_command": (
            f"ssh {settings['ssh_user']}@{node['publicIpAddress']} -p {node['port']}"
        ),
    }


def connect(node: dict[str, Any], settings: dict[str, str]) -> paramiko.SSHClient:
    STATE_DIR.mkdir(exist_ok=True)
    client = paramiko.SSHClient()
    if KNOWN_HOSTS.exists():
        client.load_host_keys(str(KNOWN_HOSTS))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=node["host"],
        port=node["port"],
        username=settings["ssh_user"],
        password=settings["ssh_password"],
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    client.save_host_keys(str(KNOWN_HOSTS))
    return client


def remote_command(
    client: paramiko.SSHClient,
    command: str,
    *,
    stdin_text: str | None = None,
    timeout: int = 900,
) -> tuple[int, str, str]:
    transport = client.get_transport()
    if transport is None:
        raise AutomationError("SSH transport is unavailable")
    channel = transport.open_session(timeout=timeout)
    channel.settimeout(timeout)
    channel.set_combine_stderr(True)
    channel.exec_command(command)
    if stdin_text is not None:
        channel.sendall(stdin_text.encode())
    channel.shutdown_write()
    output = channel.makefile("rb").read().decode(errors="replace")
    exit_code = channel.recv_exit_status()
    return exit_code, output, ""


def sudo_password(settings: dict[str, str]) -> str:
    return settings["ssh_password"] + "\n"


def node_is_healthy(node: dict[str, Any], settings: dict[str, str]) -> bool:
    try:
        with connect(node, settings) as client:
            code, _, _ = remote_command(
                client,
                "sudo -S -p '' -- bash -c "
                + shlex.quote(
                    "systemctl is-active --quiet baron-blobfuse2.service "
                    "&& systemctl is-active --quiet baron-sanity.service "
                    "&& mountpoint -q /mnt/baron-training "
                    "&& test -s /mnt/baron-training/sanity/latest.txt"
                ),
                stdin_text=sudo_password(settings),
                timeout=60,
            )
            return code == 0
    except (AutomationError, OSError, paramiko.SSHException):
        return False


def bootstrap_node(
    node: dict[str, Any], identity_client_id: str, settings: dict[str, str]
) -> None:
    if not LOCAL_BOOTSTRAP.exists():
        raise AutomationError(f"Missing bootstrap file: {LOCAL_BOOTSTRAP}")

    print(f"Connecting with dynamically discovered endpoint: {node['ssh_command']}")
    with connect(node, settings) as client:
        identity_url = (
            "http://169.254.169.254/metadata/identity/oauth2/token"
            "?api-version=2018-02-01"
            "&resource=https%3A%2F%2Fstorage.azure.com%2F"
            f"&client_id={identity_client_id}"
        )
        identity_check, _, _ = remote_command(
            client,
            "curl -fsS -o /dev/null -H Metadata:true " + shlex.quote(identity_url),
            timeout=30,
        )
        if identity_check == 0:
            print("Using baron-umi managed identity for Blob Storage")
            auth_environment = (
                f"AZURE_STORAGE_ACCOUNT={resource_name(settings['storage_account_id'])}\n"
                f"AZURE_STORAGE_ACCOUNT_CONTAINER={settings['storage_container']}\n"
                "AZURE_STORAGE_AUTH_TYPE=msi\n"
                f"AZURE_STORAGE_IDENTITY_CLIENT_ID={identity_client_id}\n"
            )
        else:
            print(
                "WARNING: baron-umi is not exposed by this AML host node; "
                "using a root-protected storage account key fallback"
            )
            storage_key = get_storage_account_key(settings)
            auth_environment = (
                f"AZURE_STORAGE_ACCOUNT={resource_name(settings['storage_account_id'])}\n"
                f"AZURE_STORAGE_ACCOUNT_CONTAINER={settings['storage_container']}\n"
                "AZURE_STORAGE_AUTH_TYPE=key\n"
                f"AZURE_STORAGE_ACCESS_KEY={storage_key}\n"
            )

        with client.open_sftp() as sftp:
            sftp.put(str(LOCAL_BOOTSTRAP), REMOTE_BOOTSTRAP)
            sftp.chmod(REMOTE_BOOTSTRAP, 0o700)
            encoded_environment = auth_environment.encode()
            sftp.putfo(
                io.BytesIO(encoded_environment),
                "/tmp/baron-storage.env",
                file_size=len(encoded_environment),
            )
            sftp.chmod("/tmp/baron-storage.env", 0o600)

        arguments = [
            REMOTE_BOOTSTRAP,
            "/tmp/baron-storage.env",
            str(node["node_id"]),
            settings["ssh_user"],
        ]
        command = "sudo -S -p '' -- " + " ".join(
            shlex.quote(value) for value in arguments
        )
        code, output, error = remote_command(
            client, command, stdin_text=sudo_password(settings), timeout=1200
        )
        if output.strip():
            print(output.strip())
        if code:
            raise AutomationError(
                f"Remote bootstrap failed: {error.strip() or output.strip()}"
            )

        verify = (
            "set -e; "
            "systemctl is-active baron-blobfuse2.service; "
            "systemctl is-active baron-sanity.service; "
            "mountpoint -q /mnt/baron-training; "
            "test -s /mnt/baron-training/sanity/latest.txt; "
            "cat /mnt/baron-training/sanity/latest.txt"
        )
        deadline = time.monotonic() + 45
        while True:
            code, output, error = remote_command(
                client,
                "sudo -S -p '' -- bash -c " + shlex.quote(verify),
                stdin_text=sudo_password(settings),
                timeout=60,
            )
            if code == 0:
                print("Verification passed:")
                print(output.strip())
                return
            if time.monotonic() >= deadline:
                _, logs, _ = remote_command(
                    client,
                    "sudo -S -p '' -- journalctl -u baron-blobfuse2 "
                    "-u baron-sanity -n 80 --no-pager",
                    stdin_text=sudo_password(settings),
                    timeout=60,
                )
                raise AutomationError(
                    f"Services did not become healthy: {error.strip()}\n{logs}"
                )
            time.sleep(5)


def stop_node(node: dict[str, Any], settings: dict[str, str]) -> None:
    print(f"Connecting with dynamically discovered endpoint: {node['ssh_command']}")
    stop_script = (
        "set -e; "
        "if systemctl cat baron-sanity.service >/dev/null 2>&1; then "
        "systemctl disable --now baron-sanity.service; fi; "
        "if systemctl cat baron-blobfuse2.service >/dev/null 2>&1; then "
        "systemctl disable --now baron-blobfuse2.service; fi; "
        "if mountpoint -q /mnt/baron-training; then "
        "fusermount3 -u /mnt/baron-training; fi; "
        "! systemctl is-active --quiet baron-sanity.service; "
        "! systemctl is-active --quiet baron-blobfuse2.service; "
        "! mountpoint -q /mnt/baron-training"
    )
    with connect(node, settings) as client:
        code, output, _ = remote_command(
            client,
            "sudo -S -p '' -- bash -c " + shlex.quote(stop_script),
            stdin_text=sudo_password(settings),
            timeout=120,
        )
    if output.strip():
        print(output.strip())
    if code:
        raise AutomationError("Remote services failed to stop cleanly")
    print("Stop verification passed: services are inactive and storage is unmounted")


def run_bootstrap_once(
    settings: dict[str, str], identity_client_id: str, previous_node: str | None
) -> str:
    node = discover_node(settings)
    print(f"Selected AML node {node['node_id']} ({node['state']})")
    if str(node["node_id"]) == previous_node and node_is_healthy(node, settings):
        print(f"Node {node['node_id']} remains healthy")
        return str(node["node_id"])
    bootstrap_node(node, identity_client_id, settings)
    return str(node["node_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage durable Blob storage on the current AML compute node."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser(
        "bootstrap", help="Install/start the mount and heartbeat services."
    )
    bootstrap.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling and bootstrap replacement nodes automatically.",
    )
    bootstrap.add_argument(
        "--interval", type=int, default=60, help="Watch polling interval in seconds."
    )
    commands.add_parser("stop", help="Stop/disable services and unmount storage.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()

    if args.command == "stop":
        node = discover_node(settings)
        print(f"Selected AML node {node['node_id']} ({node['state']})")
        stop_node(node, settings)
        return 0

    if args.interval < 10:
        raise AutomationError("--interval must be at least 10 seconds")
    ensure_storage_container(settings)
    identity_client_id = get_identity_client_id(settings)

    last_node: str | None = None
    while True:
        try:
            current_node = run_bootstrap_once(settings, identity_client_id, last_node)
            if current_node != last_node:
                print(f"Node {current_node} is configured and healthy")
            last_node = current_node
        except (AutomationError, OSError, paramiko.SSHException) as exc:
            if not args.watch:
                raise
            print(f"Watch iteration failed: {exc}", file=sys.stderr)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped", file=sys.stderr)
        raise SystemExit(130)
    except AutomationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
