FROM python:3.12-alpine

# Same system deps as your production Dockerfile
RUN apk add --no-cache \
    gcc \
    g++ \
    libffi-dev \
    openssl-dev \
    lapack-dev \
    openblas-dev \
    ffmpeg \
    python3-dev \
    musl-dev \
    bash \
    git

WORKDIR /app
