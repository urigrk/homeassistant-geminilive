ARG BUILD_FROM
FROM $BUILD_FROM

# Install system dependencies for audio processing
RUN apk add --no-cache \
    gcc \
    g++ \
    libffi-dev \
    openssl-dev \
    lapack-dev \
    openblas-dev \
    ffmpeg \
    speexdsp \
    speexdsp-dev \
    swig \
    python3-dev \
    standard-imp \
    python3 py3-pip

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY bridge.py config.py aec.py ha_client.py prompts.py session.py tools.py audio.py ./
 
# Register as an s6 service.
# The HA base image already sets ENTRYPOINT ["/init"] (s6-overlay PID 1).
# Do NOT override CMD — just drop the run script into /etc/services.d/
# and s6 will supervise it automatically.
COPY run.sh /etc/services.d/gemini_bridge/run
RUN chmod a+x /etc/services.d/gemini_bridge/run
