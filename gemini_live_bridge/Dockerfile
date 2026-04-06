ARG BUILD_FROM
FROM $BUILD_FROM

# Install system dependencies for audio processing
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    libffi-dev \
    openssl-dev \
    lapack-dev \
    openblas-dev \
    ffmpeg \
    python3 python3-pip

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY bridge.py .
COPY run.sh .
RUN chmod +x run.sh

CMD ["/app/run.sh"]
