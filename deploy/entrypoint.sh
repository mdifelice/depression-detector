#!/usr/bin/env sh
set -e

PORT="${PORT:-7860}"
sed "s/__PORT__/${PORT}/g" /etc/nginx/conf.d/default.conf > /tmp/nginx.conf
mv /tmp/nginx.conf /etc/nginx/conf.d/default.conf

exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/app.conf