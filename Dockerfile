FROM python:3.12-slim

WORKDIR /opt

# git for pulling, tzdata so ZoneInfo/TZ work properly
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates tzdata \
  && rm -rf /var/lib/apt/lists/*

# Bot deps (discord + aiohttp)
RUN pip install --no-cache-dir -U discord.py aiohttp

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV REPO_URL="https://github.com/hmxmilohax/GoBot"
ENV APP_DIR="/opt/gobot"
ENV BRANCH="main"

ENTRYPOINT ["/entrypoint.sh"]
