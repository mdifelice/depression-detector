#!/usr/bin/env sh
set -e

if [ ! -f .env ]; then
  printf 'DISABLE_AUTH=true\n' > .env
fi

docker compose up --build -d