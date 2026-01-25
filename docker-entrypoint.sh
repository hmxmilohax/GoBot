#!/bin/sh
set -eu

# config.ini from host
if [ ! -f /config/config.ini ]; then
  echo "[gobot] ERROR: /config/config.ini not found (bind-mount it)."
  exit 1
fi
cp /config/config.ini /app/config.ini

# Persist state file directly on host via bind mount
mkdir -p /data
if [ ! -f /data/battle_manager_state.json ]; then
  echo "{}" > /data/battle_manager_state.json
fi
ln -sf /data/battle_manager_state.json /app/battle_manager_state.json

cd /app
exec python GoBot.py
