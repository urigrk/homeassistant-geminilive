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
    swig \
    python3-dev \
    python3 py3-pip

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
RUN pip3 download --no-deps --no-cache-dir webrtc-audio-processing -d /tmp/wap && \
    cd /tmp/wap && \
    tar xf webrtc-audio-processing-*.tar.gz && \
    sed -i 's/#include <string>/#include <string>\n#include <cstdint>/' \
        webrtc-audio-processing-*/src/audio_processing_module.h && \
    pip3 install --no-cache-dir --break-system-packages /tmp/wap/webrtc-audio-processing-*/


COPY bridge.py .
 
# Register as an s6 service.
# The HA base image already sets ENTRYPOINT ["/init"] (s6-overlay PID 1).
# Do NOT override CMD — just drop the run script into /etc/services.d/
# and s6 will supervise it automatically.
COPY run.sh /etc/services.d/gemini_bridge/run
RUN chmod a+x /etc/services.d/gemini_bridge/run
