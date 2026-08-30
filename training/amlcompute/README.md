# AML compute durable storage runner

This automation dynamically obtains the active node IP and port from the Azure
Machine Learning `listNodes` REST response, connects over SSH, mounts the
configured Blob container, and installs a background sanity heartbeat.

Authentication uses the configured user-assigned managed identity when AML
exposes it to the host through IMDS. AML cluster host nodes may expose identities
only to submitted job containers; in that case the bootstrap retrieves a storage
account key and stores it in `/etc/baron-storage.env` with root-only permissions.
The SSH password remains in the git-ignored local `.env` file as requested.

## Prerequisites

- Azure CLI authenticated to the Microsoft tenant and authorized for both subscriptions.
- Python 3.10 or newer.
- Dependencies installed from this directory's `requirements.txt`.
- `.env` populated using `.env.example` as its template.

## Bootstrap

From the repository root:

    python -m pip install -r training/amlcompute/requirements.txt
    python training/amlcompute/amlrunner.py bootstrap

Bootstrap is idempotent and convergent:

- The Blob container `PUT` creates or updates the same container.
- BlobFuse2 is installed only when absent.
- Directories are safely recreated with `install -d`.
- Scripts, credentials, and systemd unit definitions are replaced with the desired content.
- `systemctl enable` is safe when already enabled.
- Services are restarted and health-checked, so configuration changes take effect.
- Existing durable blobs are not deleted or overwritten, except `sanity/latest.txt`.

Running bootstrap again can briefly interrupt the mount because it intentionally
restarts the services, but it does not create duplicate services or mounts.

## Handle replacement nodes continuously

    python training/amlcompute/amlrunner.py bootstrap --watch --interval 60

Watch mode re-queries AML every minute. It leaves a healthy current node alone
and idempotently bootstraps a new or unhealthy node. It must run on an always-on
controller because an evicted AML node cannot bootstrap its replacement.

## Stop

    python training/amlcompute/amlrunner.py stop

Stop dynamically locates the current node, stops and disables both services,
unmounts storage, and verifies the stopped state. Calling stop again is safe.
It does not uninstall BlobFuse2, remove unit definitions, delete credentials, or
delete anything in Blob Storage; a later bootstrap starts everything again.

## Remote paths and logs

- Mounted Blob container: `/mnt/baron-training`
- Durable latest heartbeat: `/mnt/baron-training/sanity/latest.txt`
- Durable heartbeat history: `/mnt/baron-training/sanity/logs/`
- Mount service logs: `journalctl -u baron-blobfuse2`
- Sanity service logs: `journalctl -u baron-sanity`

Training should write completed, versioned checkpoints beneath the mounted path.
For large checkpoints, write locally first and move or copy the completed artifact
into the mount; BlobFuse is not a fully POSIX-compatible filesystem.