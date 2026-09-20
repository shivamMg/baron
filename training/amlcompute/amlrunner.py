#!/usr/bin/env python3
"""Manage durable Blob storage services on an Azure ML compute node."""

from __future__ import annotations

import argparse
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path, PurePosixPath
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
REMOTE_TRAINING_SCRIPT = "/tmp/run-training.sh"
LOCAL_TRAINING_SCRIPT = BASE_DIR / "remote" / "run-training.sh"
REMOTE_TRAINING_SERVICE = "/tmp/baron-training.service"
LOCAL_TRAINING_SERVICE = BASE_DIR / "remote" / "baron-training.service"
STATE_DIR = BASE_DIR / ".state"
KNOWN_HOSTS = STATE_DIR / "known_hosts"
LOG_FILE = STATE_DIR / "amlrunner.log"
SELECTED_NODE_FILE = STATE_DIR / "selected-node.json"

AML_API_VERSION = "2024-10-01"
STORAGE_API_VERSION = "2023-05-01"
LOGGER = logging.getLogger("baron.amlrunner")


class AutomationError(RuntimeError):
    pass


def configure_logging(log_path: Path = LOG_FILE) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AutomationError(f"Missing required setting {name} in {ENV_FILE}")
    return value


def parse_compute_resource_ids(value: str) -> list[str]:
    resource_ids = [item.strip().rstrip("/") for item in value.split(",")]
    resource_ids = [item for item in resource_ids if item]
    if not resource_ids:
        raise AutomationError("No AML compute resource IDs were configured")
    return list(dict.fromkeys(resource_ids))


def resolve_run_id(training_enabled: str) -> str:
    run_id = os.environ.get("BARON_RUN_ID", "").strip()
    if run_id:
        return run_id
    if training_enabled == "true":
        raise AutomationError(f"Missing required setting BARON_RUN_ID in {ENV_FILE}")
    LOGGER.info("BARON_RUN_ID is not set; using run_id=dry-run")
    return "dry-run"


def load_settings() -> dict[str, Any]:
    load_dotenv(ENV_FILE)
    ssh_user = required_env("AML_SSH_USER")
    compute_ids = required_env("AML_COMPUTE_RESOURCE_IDS")
    training_enabled = os.environ.get("BARON_TRAINING_ENABLED", "false").strip().lower()
    if training_enabled not in {"true", "false"}:
        raise AutomationError("BARON_TRAINING_ENABLED must be true or false")
    return {
        "aml_compute_ids": parse_compute_resource_ids(compute_ids),
        "storage_account_id": required_env("STORAGE_ACCOUNT_RESOURCE_ID").rstrip("/"),
        "storage_container": required_env("STORAGE_CONTAINER"),
        "ssh_user": ssh_user,
        "ssh_password": required_env("AML_SSH_PASSWORD"),
        "run_id": resolve_run_id(training_enabled),
        "durable_path": os.environ.get(
            "BARON_DURABLE_PATH", "/mnt/baron-training"
        ).strip(),
        "local_path": os.environ.get("BARON_LOCAL_PATH", "/mnt/baron-local").strip(),
        "service_user": os.environ.get("BARON_SERVICE_USER", ssh_user).strip(),
        "training_enabled": training_enabled,
        "dry_run_steps": os.environ.get("BARON_DRY_RUN_STEPS", "3").strip(),
        "dry_run_interval": os.environ.get(
            "BARON_DRY_RUN_INTERVAL_SECONDS", "1"
        ).strip(),
        "image": os.environ.get("BARON_IMAGE", "").strip(),
        "config_path": os.environ.get("BARON_CONFIG_PATH", "").strip(),
        "config_hash": os.environ.get("BARON_CONFIG_HASH", "").strip(),
        "progress_timeout": os.environ.get(
            "BARON_PROGRESS_TIMEOUT_SECONDS", "600"
        ).strip(),
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


def ensure_storage_container(settings: dict[str, Any]) -> None:
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


def get_storage_account_key(settings: dict[str, Any]) -> str:
    keys = az_json(
        "rest",
        "--method", "post",
        "--url",
        arm_url(f"{settings['storage_account_id']}/listKeys", STORAGE_API_VERSION),
    ).get("keys", [])
    if not keys or not keys[0].get("value"):
        raise AutomationError("Storage account listKeys returned no usable key")
    return keys[0]["value"]


def discover_nodes(settings: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    errors = []
    for compute_id in settings["aml_compute_ids"]:
        compute_name = resource_name(compute_id)
        try:
            response = az_json(
                "rest",
                "--method",
                "post",
                "--url",
                arm_url(f"{compute_id}/listNodes", AML_API_VERSION),
            )
        except AutomationError as exc:
            errors.append(f"{compute_name}: {exc}")
            LOGGER.warning("AML node discovery failed compute=%s error=%s", compute_name, exc)
            continue
        nodes = response.get("nodes", response.get("value", response))
        if isinstance(nodes, dict):
            nodes = [nodes]
        usable = [
            node for node in nodes or []
            if node.get("publicIpAddress") and node.get("port")
        ]
        LOGGER.info(
            "discovered AML nodes compute=%s total=%s usable=%s",
            compute_name,
            len(nodes or []),
            len(usable),
        )
        for node in usable:
            candidates.append(
                {
                    "compute_id": compute_id,
                    "compute_name": compute_name,
                    "node_id": str(node.get("nodeId", "unknown-node")),
                    "state": str(node.get("nodeState", "unknown")).lower(),
                    "host": node["publicIpAddress"],
                    "port": int(node["port"]),
                    "ssh_command": (
                        f"ssh {settings['ssh_user']}@{node['publicIpAddress']} "
                        f"-p {node['port']}"
                    ),
                }
            )
    if not candidates and errors and len(errors) == len(settings["aml_compute_ids"]):
        raise AutomationError("AML node discovery failed for every compute: " + "; ".join(errors))
    return candidates


def load_selected_node(path: Path = SELECTED_NODE_FILE) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("ignoring invalid selected-node state path=%s error=%s", path, exc)
        return None
    if not isinstance(value, dict) or not all(
        isinstance(value.get(key), str) for key in ("compute_id", "node_id")
    ):
        LOGGER.warning("ignoring invalid selected-node state path=%s", path)
        return None
    return {"compute_id": value["compute_id"], "node_id": value["node_id"]}


def save_selected_node(node: dict[str, Any], path: Path = SELECTED_NODE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"compute_id": node["compute_id"], "node_id": str(node["node_id"])},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def find_selected_node(
    candidates: list[dict[str, Any]], selection: dict[str, str] | None
) -> dict[str, Any] | None:
    if selection is None:
        return None
    return next(
        (
            node
            for node in candidates
            if node["compute_id"] == selection["compute_id"]
            and str(node["node_id"]) == selection["node_id"]
        ),
        None,
    )


def connect(node: dict[str, Any], settings: dict[str, Any]) -> paramiko.SSHClient:
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


def sudo_password(settings: dict[str, Any]) -> str:
    return settings["ssh_password"] + "\n"


def put_remote_bytes(
    sftp: paramiko.SFTPClient, data: bytes, remote_path: str, mode: int
) -> None:
    """Upload bytes, applying the target mode before any content is written."""
    sftp.open(remote_path, "wb").close()
    sftp.chmod(remote_path, mode)
    sftp.putfo(io.BytesIO(data), remote_path, file_size=len(data))


def put_remote_script(
    sftp: paramiko.SFTPClient, local_path: Path, remote_path: str, mode: int
) -> None:
    """Upload a text file with LF endings so the node can execute it."""
    data = local_path.read_bytes().replace(b"\r\n", b"\n")
    put_remote_bytes(sftp, data, remote_path, mode)


def training_environment(settings: dict[str, Any]) -> str:
    values = {
        "BARON_RUN_ID": settings["run_id"],
        "BARON_DURABLE_PATH": settings["durable_path"],
        "BARON_LOCAL_PATH": settings["local_path"],
        "BARON_TRAINING_ENABLED": settings["training_enabled"],
        "BARON_DRY_RUN_STEPS": settings["dry_run_steps"],
        "BARON_DRY_RUN_INTERVAL_SECONDS": settings["dry_run_interval"],
        "BARON_IMAGE": settings["image"],
        "BARON_CONFIG_PATH": settings["config_path"],
        "BARON_CONFIG_HASH": settings["config_hash"],
    }
    if any("\n" in value or "\r" in value for value in values.values()):
        raise AutomationError("Training settings must not contain newlines")
    return "".join(f"{name}={shlex.quote(value)}\n" for name, value in values.items())


def parse_training_observation(
    service_state: str, status_text: str, metric_text: str
) -> dict[str, Any]:
    status: dict[str, Any] = {}
    metric: dict[str, Any] = {}
    for text, target in ((status_text, status), (metric_text, metric)):
        if text.strip():
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                target.update(value)
    step = status.get("step")
    if type(step) is not int:
        step = metric.get("step")
    if type(step) is not int:
        step = None
    return {
        "installed": service_state != "missing",
        "active": service_state == "active",
        "status": status.get("status"),
        "step": step,
        "dry_run": status.get("dry_run") is True,
    }


def evaluate_training_progress(
    observation: dict[str, Any],
    previous_step: int | None,
    last_progress_at: float,
    now: float,
    timeout: int,
) -> tuple[int | None, float, str]:
    if observation["status"] in {"completed", "fatal"}:
        return previous_step, last_progress_at, "terminal"
    if not observation["active"]:
        return previous_step, last_progress_at, "inactive"
    step = observation["step"]
    if step is None:
        event = "stalled" if now - last_progress_at >= timeout else "waiting"
        return previous_step, now if event == "stalled" else last_progress_at, event
    if previous_step is None:
        return step, now, "observed"
    if step > previous_step:
        return step, now, "advanced"
    if step < previous_step:
        return previous_step, last_progress_at, "regressed"
    if now - last_progress_at >= timeout:
        return previous_step, now, "stalled"
    return previous_step, last_progress_at, "waiting"


def observe_training(
    client: paramiko.SSHClient,
    settings: dict[str, str],
    *,
    read_progress: bool = True,
) -> dict[str, Any]:
    state_script = (
        "if ! systemctl cat baron-training.service >/dev/null 2>&1; then "
        "printf missing; elif systemctl is-active --quiet baron-training.service; "
        "then printf active; else printf inactive; fi"
    )
    _, service_state, _ = remote_command(
        client,
        "sudo -S -p '' -- bash -c " + shlex.quote(state_script),
        stdin_text=sudo_password(settings),
        timeout=60,
    )
    if not read_progress:
        return parse_training_observation(service_state.strip(), "", "")
    status_path = (
        PurePosixPath(settings["durable_path"])
        / "runs"
        / settings["run_id"]
        / "status.json"
    )
    _, status_text, _ = remote_command(
        client,
        "sudo -S -p '' -- bash -c "
        + shlex.quote(
            f"if test -s {shlex.quote(str(status_path))}; then "
            f"cat {shlex.quote(str(status_path))}; fi"
        ),
        stdin_text=sudo_password(settings),
        timeout=60,
    )
    log_dir = PurePosixPath(settings["local_path"]) / "logs" / settings["run_id"]
    metric_script = (
        f"latest=$(find {shlex.quote(str(log_dir))} -maxdepth 1 -type f "
        "-name '*.json' -print 2>/dev/null | sort | tail -n 1); "
        'if test -n "$latest"; then cat "$latest"; fi'
    )
    _, metric_text, _ = remote_command(
        client,
        "sudo -S -p '' -- bash -c " + shlex.quote(metric_script),
        stdin_text=sudo_password(settings),
        timeout=60,
    )
    return parse_training_observation(
        service_state.strip(), status_text.strip(), metric_text.strip()
    )


def training_journal(node: dict[str, Any], settings: dict[str, Any]) -> str:
    with connect(node, settings) as client:
        _, output, _ = remote_command(
            client,
            "sudo -S -p '' -- journalctl -u baron-training.service "
            "-n 80 --no-pager",
            stdin_text=sudo_password(settings),
            timeout=60,
        )
    return output.strip()


def inspect_node(
    node: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    with connect(node, settings) as client:
        code, _, _ = remote_command(
            client,
            "sudo -S -p '' -- bash -c "
            + shlex.quote(
                "systemctl is-active --quiet baron-blobfuse2.service "
                f"&& mountpoint -q {shlex.quote(settings['durable_path'])}"
            ),
            stdin_text=sudo_password(settings),
            timeout=60,
        )
        storage_healthy = code == 0
        owns_run_code, _, _ = remote_command(
            client,
            "sudo -S -p '' -- bash -c "
            + shlex.quote(
                "test -r /etc/baron-training.env && "
                ". /etc/baron-training.env && "
                f"test \"$BARON_RUN_ID\" = {shlex.quote(settings['run_id'])}"
            ),
            stdin_text=sudo_password(settings),
            timeout=60,
        )
        observation = observe_training(
            client, settings, read_progress=storage_healthy
        )
    return {
        "storage_healthy": storage_healthy,
        "owns_run": owns_run_code == 0,
        "training": observation,
    }


def bootstrap_node(
    node: dict[str, Any], storage_account_key: str, settings: dict[str, Any]
) -> None:
    for path in (LOCAL_BOOTSTRAP, LOCAL_TRAINING_SCRIPT, LOCAL_TRAINING_SERVICE):
        if not path.exists():
            raise AutomationError(f"Missing bootstrap file: {path}")

    LOGGER.info(
        "connecting to AML node compute=%s node=%s",
        node["compute_name"],
        node["node_id"],
    )
    with connect(node, settings) as client:
        LOGGER.info("using a root-protected storage account key for BlobFuse2")
        auth_environment = (
            f"AZURE_STORAGE_ACCOUNT={resource_name(settings['storage_account_id'])}\n"
            f"AZURE_STORAGE_ACCOUNT_CONTAINER={settings['storage_container']}\n"
            "AZURE_STORAGE_AUTH_TYPE=key\n"
            f"AZURE_STORAGE_ACCESS_KEY={storage_account_key}\n"
        )
        runtime_environment = training_environment(settings).encode()

        with client.open_sftp() as sftp:
            put_remote_script(sftp, LOCAL_BOOTSTRAP, REMOTE_BOOTSTRAP, 0o700)
            put_remote_script(
                sftp, LOCAL_TRAINING_SCRIPT, REMOTE_TRAINING_SCRIPT, 0o700
            )
            put_remote_script(
                sftp, LOCAL_TRAINING_SERVICE, REMOTE_TRAINING_SERVICE, 0o600
            )
            put_remote_bytes(
                sftp, auth_environment.encode(), "/tmp/baron-storage.env", 0o600
            )
            put_remote_bytes(
                sftp, runtime_environment, "/tmp/baron-training.env", 0o600
            )

        arguments = [
            REMOTE_BOOTSTRAP,
            "/tmp/baron-storage.env",
            str(node["node_id"]),
            settings["service_user"],
            settings["durable_path"],
            settings["local_path"],
        ]
        command = "sudo -S -p '' -- " + " ".join(
            shlex.quote(value) for value in arguments
        )
        code, output, error = remote_command(
            client, command, stdin_text=sudo_password(settings), timeout=1200
        )
        if output.strip():
            LOGGER.info("remote bootstrap output:\n%s", output.strip())
        if code:
            raise AutomationError(
                f"Remote bootstrap failed: {error.strip() or output.strip()}"
            )

        verify = (
            "set -e; "
            "systemctl is-active baron-blobfuse2.service; "
            f"mountpoint -q {shlex.quote(settings['durable_path'])}"
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
                LOGGER.info("bootstrap verification passed: %s", output.strip())
                return
            if time.monotonic() >= deadline:
                _, logs, _ = remote_command(
                    client,
                    "sudo -S -p '' -- journalctl -u baron-blobfuse2 "
                    "-n 80 --no-pager",
                    stdin_text=sudo_password(settings),
                    timeout=60,
                )
                raise AutomationError(
                    f"Services did not become healthy: {error.strip()}\n{logs}"
                )
            time.sleep(5)


def stop_node(node: dict[str, Any], settings: dict[str, Any]) -> None:
    LOGGER.info(
        "connecting to AML node for stop compute=%s node=%s",
        node["compute_name"],
        node["node_id"],
    )
    durable_path = shlex.quote(settings["durable_path"])
    stop_script = (
        "set -e; "
        "if systemctl cat baron-training.service >/dev/null 2>&1; then "
        "systemctl disable --now baron-training.service; fi; "
        "docker rm -f baron-training >/dev/null 2>&1 || true; "
        "if systemctl cat baron-blobfuse2.service >/dev/null 2>&1; then "
        "systemctl disable --now baron-blobfuse2.service; fi; "
        f"if mountpoint -q {durable_path}; then "
        f"fusermount3 -u {durable_path}; fi; "
        "! systemctl is-active --quiet baron-training.service; "
        "! systemctl is-active --quiet baron-blobfuse2.service; "
        f"! mountpoint -q {durable_path}"
    )
    with connect(node, settings) as client:
        code, output, _ = remote_command(
            client,
            "sudo -S -p '' -- bash -c " + shlex.quote(stop_script),
            stdin_text=sudo_password(settings),
            timeout=120,
        )
    if output.strip():
        LOGGER.info("remote stop output:\n%s", output.strip())
    if code:
        raise AutomationError("Remote services failed to stop cleanly")
    LOGGER.info("stop verification passed: storage service inactive and unmounted")


def reconcile_once(
    settings: dict[str, Any], previous_node: dict[str, str] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = discover_nodes(settings)
    previous = find_selected_node(candidates, previous_node)
    inspection = None
    if previous is not None:
        LOGGER.info(
            "inspecting previous AML node compute=%s node=%s state=%s",
            previous["compute_name"],
            previous["node_id"],
            previous["state"],
        )
        try:
            previous_inspection = inspect_node(previous, settings)
        except (OSError, paramiko.SSHException) as exc:
            LOGGER.warning(
                "previous AML node is unreachable compute=%s node=%s error=%s",
                previous["compute_name"],
                previous["node_id"],
                exc,
            )
        else:
            if previous_inspection["owns_run"]:
                node = previous
                inspection = previous_inspection
                LOGGER.info(
                    "reusing previous AML node compute=%s node=%s state=%s",
                    node["compute_name"],
                    node["node_id"],
                    node["state"],
                )
            else:
                LOGGER.warning(
                    "previous AML node does not own run compute=%s node=%s run_id=%s",
                    previous["compute_name"],
                    previous["node_id"],
                    settings["run_id"],
                )
    elif previous_node is not None:
        LOGGER.warning(
            "previous AML node is absent compute=%s node=%s",
            resource_name(previous_node["compute_id"]),
            previous_node["node_id"],
        )
    if inspection is None:
        idle = [candidate for candidate in candidates if candidate["state"] == "idle"]
        if not idle:
            raise AutomationError("No previous run node or Idle AML node is available")
        node = idle[0]
        LOGGER.info(
            "selected Idle AML node compute=%s node=%s state=%s",
            node["compute_name"],
            node["node_id"],
            node["state"],
        )
        inspection = inspect_node(node, settings)
    observation = inspection["training"]
    if inspection["storage_healthy"] and inspection["owns_run"]:
        if observation["active"] or observation["status"] in {"completed", "fatal"}:
            LOGGER.info(
                "AML node remains healthy compute=%s node=%s",
                node["compute_name"],
                node["node_id"],
            )
            return node, observation
        LOGGER.warning(
            "training service is missing or unexpectedly inactive; bootstrapping"
        )
    elif not inspection["storage_healthy"]:
        LOGGER.warning("durable storage mount is unhealthy; bootstrapping")
    else:
        LOGGER.warning(
            "AML node does not own run compute=%s node=%s run_id=%s; "
            "bootstrapping to take ownership",
            node["compute_name"],
            node["node_id"],
            settings["run_id"],
        )
    ensure_storage_container(settings)
    storage_account_key = get_storage_account_key(settings)
    bootstrap_node(node, storage_account_key, settings)
    # Persist the selection before the readiness assertions below so a failed
    # check re-targets this node instead of bootstrapping a second one.
    save_selected_node(node)
    if previous_node is not None and (
        node["compute_id"] != previous_node["compute_id"]
        or str(node["node_id"]) != previous_node["node_id"]
    ):
        LOGGER.info(
            "replacement AML node recovered compute=%s node=%s",
            node["compute_name"],
            node["node_id"],
        )
    inspection = inspect_node(node, settings)
    observation = inspection["training"]
    if not inspection["storage_healthy"]:
        raise AutomationError("durable storage mount remained unhealthy after bootstrap")
    if not inspection["owns_run"]:
        raise AutomationError("baron-training.service does not own the configured run")
    if not observation["installed"]:
        raise AutomationError("baron-training.service was not installed by bootstrap")
    if (
        not observation["active"]
        and observation["status"] not in {"completed", "fatal"}
    ):
        raise AutomationError("baron-training.service did not become active")
    return node, observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage durable Blob storage on the current AML compute node."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Inspect and reconcile the current AML node continuously."
    )
    run.add_argument(
        "--once",
        action="store_true",
        help="Perform one inspection and reconciliation, then exit.",
    )
    run.add_argument(
        "--interval", type=int, default=60, help="Watch polling interval in seconds."
    )
    commands.add_parser("stop", help="Stop/disable services and unmount storage.")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    settings = load_settings()
    LOGGER.info("AMLRunner started command=%s", args.command)

    if args.command == "stop":
        selection = load_selected_node()
        node = find_selected_node(discover_nodes(settings), selection)
        if node is None:
            raise AutomationError("The previously selected AML node is unavailable")
        LOGGER.info(
            "selected AML node for stop compute=%s node=%s state=%s",
            node["compute_name"],
            node["node_id"],
            node["state"],
        )
        stop_node(node, settings)
        return 0

    if args.interval < 10:
        raise AutomationError("--interval must be at least 10 seconds")

    last_node = load_selected_node()
    last_step: int | None = None
    last_progress_at = time.monotonic()
    try:
        progress_timeout = int(settings["progress_timeout"])
    except ValueError as exc:
        raise AutomationError("BARON_PROGRESS_TIMEOUT_SECONDS must be an integer") from exc
    if progress_timeout <= 0:
        raise AutomationError("BARON_PROGRESS_TIMEOUT_SECONDS must be positive")
    while True:
        try:
            current_node, observation = reconcile_once(settings, last_node)
            current_selection = {
                "compute_id": current_node["compute_id"],
                "node_id": str(current_node["node_id"]),
            }
            if current_selection != last_node:
                LOGGER.info(
                    "AML node is configured and healthy compute=%s node=%s",
                    current_node["compute_name"],
                    current_node["node_id"],
                )
                last_step = None
                last_progress_at = time.monotonic()
            save_selected_node(current_node)
            now = time.monotonic()
            last_step, last_progress_at, progress_event = evaluate_training_progress(
                observation,
                last_step,
                last_progress_at,
                now,
                progress_timeout,
            )
            if progress_event == "terminal":
                LOGGER.info("training reached terminal status=%s", observation["status"])
            elif progress_event == "observed":
                LOGGER.info("observed training step=%s", observation["step"])
            elif progress_event == "advanced":
                LOGGER.info(
                    "confirmed training progress step=%s dry_run=%s",
                    observation["step"],
                    observation["dry_run"],
                )
            elif progress_event == "regressed":
                LOGGER.error(
                    "training step regressed previous_step=%s observed_step=%s",
                    last_step,
                    observation["step"],
                )
            elif progress_event == "stalled":
                LOGGER.error(
                    "training progress stalled step=%s timeout_seconds=%s\n%s",
                    observation["step"],
                    progress_timeout,
                    training_journal(current_node, settings),
                )
            last_node = current_selection
        except (AutomationError, OSError, paramiko.SSHException) as exc:
            if args.once:
                raise
            LOGGER.exception("watch iteration failed: %s", exc)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.info("AMLRunner stopped by operator")
        raise SystemExit(130)
    except AutomationError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1)
