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
    swig \
    python3-dev \
    musl-dev \
    python3 py3-pip

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# webrtc-audio-processing 0.1.3 is missing `#include <cstdint>` in its
# vendored header, which breaks builds on modern GCC. Patch the sdist
# before installing.
RUN mkdir -p /tmp/wap \
 && pip3 download --no-deps --no-binary=:all: --dest /tmp/wap webrtc-audio-processing==0.1.3 \
 && tar -xzf /tmp/wap/webrtc?audio?processing-0.1.3.tar.gz -C /tmp/wap \
 && sed -i 's|#include <string>|#include <string>\n#include <cstdint>|' \
        /tmp/wap/webrtc_audio_processing-0.1.3/src/audio_processing_module.h \
 && sed -i 's|defined(__GLIBCXX__) && !defined(__UCLIBC__)|0|g' \
        /tmp/wap/webrtc_audio_processing-0.1.3/webrtc-audio-processing/webrtc/base/checks.cc \
 && pip3 install --no-cache-dir --break-system-packages /tmp/wap/webrtc_audio_processing-0.1.3 \
 && rm -rf /tmp/wap

COPY bridge.py config.py aec.py ha_client.py prompts.py session.py tools.py audio.py ./
 
# Register as an s6 service.
# The HA base image already sets ENTRYPOINT ["/init"] (s6-overlay PID 1).
# Do NOT override CMD — just drop the run script into /etc/services.d/
# and s6 will supervise it automatically.
COPY run.sh /etc/services.d/gemini_bridge/run
RUN chmod a+x /etc/services.d/gemini_bridge/run
