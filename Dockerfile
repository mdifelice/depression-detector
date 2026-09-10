# --- frontend build stage ---
FROM node:22-slim AS frontend

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ .
ARG VITE_API_URL=""
RUN npm run build

# --- backend dependencies stage (cache pip layer) ---
FROM python:3.13-slim AS backend

WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- runtime: nginx + FastAPI behind it ---
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx supervisor curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=backend /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

COPY backend/ /app/backend/
COPY --from=frontend /app/dist /usr/share/nginx/html
COPY space/nginx.conf /etc/nginx/conf.d/default.conf
COPY space/supervisord.conf /etc/supervisor/conf.d/app.conf

RUN rm -f /etc/nginx/sites-enabled/default && \
    mkdir -p /data /models /var/log/nginx

ENV DATA_DIR=/data \
    MODELS_DIR=/models \
    DISABLE_AUTH=true \
    PYTHONPATH=/app/backend

EXPOSE 7860

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/app.conf"]