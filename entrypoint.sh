#!/bin/sh
set -eu

: "${REPO_URL:=https://github.com/hmxmilohax/GoBot}"
: "${APP_DIR:=/opt/gobot}"
: "${BRANCH:=main}"

mkdir -p "$APP_DIR"
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

# battle_manager_state.json (persisted)
if [ -f /config/battle_manager_state.json ]; then
  cp /config/battle_manager_state.json "$APP_DIR/battle_manager_state.json"
else
  # If it doesn't exist yet, create an empty json so the bot can start
  echo "{}" > "$APP_DIR/battle_manager_state.json"
  cp "$APP_DIR/battle_manager_state.json" /config/battle_manager_state.json || true
fi

cd "$APP_DIR"
exec python GoBot.py
