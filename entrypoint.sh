#!/bin/sh
set -eu

: "${REPO_URL:=https://github.com/hmxmilohax/GoBot}"
: "${APP_DIR:=/opt/gobot}"
: "${BRANCH:=main}"

mkdir -p "$APP_DIR"
mkdir -p /config

git config --global --add safe.directory "$APP_DIR" || true

if [ ! -d "$APP_DIR/.git" ]; then
  echo "[gobot] Cloning repo..."
  find "$APP_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
  echo "[gobot] Updating repo..."
  git -C "$APP_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$BRANCH"
fi

# config.ini
if [ ! -f /config/config.ini ]; then
  echo "[gobot] ERROR: /config/config.ini not found (bind-mount it)."
  exit 1
fi
cp /config/config.ini "$APP_DIR/config.ini"

# battle_manager_state.json (persisted via symlink)
if [ ! -f /config/battle_manager_state.json ]; then
  echo "{}" > /config/battle_manager_state.json
fi
ln -sf /config/battle_manager_state.json "$APP_DIR/battle_manager_state.json"

cd "$APP_DIR"
exec python GoBot.py
