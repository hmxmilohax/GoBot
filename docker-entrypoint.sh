#!/bin/sh
set -eu

# config.ini from host
if [ ! -f /config/config.ini ]; then
  echo "[gobot] ERROR: /config/config.ini not found (bind-mount it)."
  exit 1
fi
cp /config/config.ini /app/config.ini

cd /app
exec python GoBot.py
