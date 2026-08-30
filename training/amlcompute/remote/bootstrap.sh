#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: bootstrap.sh AUTH_ENV_FILE NODE_ID USER" >&2
  exit 2
fi

AUTH_ENV_FILE="$1"
NODE_ID="$2"
SERVICE_USER="$3"
MOUNT_PATH="/mnt/baron-training"
CACHE_PATH="/var/cache/blobfuse2/baron-training"

if [[ ! -f "$AUTH_ENV_FILE" ]]; then
  echo "Storage authentication environment file is missing" >&2
  exit 1
fi
install -o root -g root -m 0600 "$AUTH_ENV_FILE" /etc/baron-storage.env
rm -f "$AUTH_ENV_FILE"

if ! command -v blobfuse2 >/dev/null 2>&1; then
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "Automatic BlobFuse2 installation currently supports Ubuntu; found ${ID:-unknown}" >&2
    exit 1
  fi
  apt-get update
  apt-get install -y ca-certificates curl fuse3
  curl -fsSL "https://packages.microsoft.com/config/ubuntu/${VERSION_ID}/packages-microsoft-prod.deb" \
    -o /tmp/packages-microsoft-prod.deb
  dpkg -i /tmp/packages-microsoft-prod.deb
  apt-get update
  apt-get install -y blobfuse2
fi

install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0755 "$MOUNT_PATH" "$CACHE_PATH"
grep -qxF user_allow_other /etc/fuse.conf || echo user_allow_other >> /etc/fuse.conf

# A previous service restart can leave heartbeat files in the local directory
# after FUSE has unmounted. Stop the writer first and clean only that stale path.
systemctl stop baron-sanity.service 2>/dev/null || true
systemctl stop baron-blobfuse2.service 2>/dev/null || true
if mountpoint -q "$MOUNT_PATH"; then
  fusermount3 -u "$MOUNT_PATH"
fi
rm -rf "$MOUNT_PATH/sanity"
if find "$MOUNT_PATH" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Refusing to mount over non-empty local directory: $MOUNT_PATH" >&2
  exit 1
fi

cat >/usr/local/bin/baron-sanity-heartbeat <<'HEARTBEAT'
#!/usr/bin/env bash
set -Eeuo pipefail
MOUNT_PATH=/mnt/baron-training
NODE_ID="${BARON_NODE_ID:?}"
mkdir -p "$MOUNT_PATH/sanity/logs"
while true; do
  timestamp="$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
  line="timestamp=$timestamp node=$NODE_ID host=$(hostname) status=ok"
  printf '%s\n' "$line" >"$MOUNT_PATH/sanity/latest.txt"
  printf '%s\n' "$line" >"$MOUNT_PATH/sanity/logs/${timestamp//:/-}-${NODE_ID}.log"
  echo "$line"
  sleep 60
done
HEARTBEAT
chmod 0755 /usr/local/bin/baron-sanity-heartbeat

cat >/etc/systemd/system/baron-blobfuse2.service <<EOF
[Unit]
Description=Baron training Azure Blob Storage mount
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
EnvironmentFile=/etc/baron-storage.env
ExecStart=/usr/bin/blobfuse2 mount $MOUNT_PATH --tmp-path=$CACHE_PATH --allow-other --foreground=true --log-type=syslog --log-level=LOG_WARNING
ExecStop=/bin/fusermount3 -u $MOUNT_PATH
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

cat >/etc/systemd/system/baron-sanity.service <<EOF
[Unit]
Description=Baron durable-storage sanity heartbeat
Requires=baron-blobfuse2.service
After=baron-blobfuse2.service

[Service]
Type=simple
User=$SERVICE_USER
Environment=BARON_NODE_ID=$NODE_ID
ExecStartPre=/usr/bin/mountpoint -q $MOUNT_PATH
ExecStart=/usr/local/bin/baron-sanity-heartbeat
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable baron-blobfuse2.service baron-sanity.service
systemctl start baron-blobfuse2.service

for _ in {1..30}; do
  mountpoint -q "$MOUNT_PATH" && break
  sleep 1
done
mountpoint -q "$MOUNT_PATH"
systemctl start baron-sanity.service

echo "BlobFuse2 and sanity services installed and started."