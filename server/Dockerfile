# syntax=docker/dockerfile:1
#
# VietASR server image — slim. Build context MUST be the repo root:
#   docker build -f server/Dockerfile -t vietasr-server .
# or just `docker compose up`.

# ---- builder: compile libvietasr, install everything into a venv, slim it --
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY core/ core/
COPY models/ models/
COPY bindings/python/ bindings/python/

# Build the native binding wheel (libvietasr + onnxruntime + model embedded).
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /wheels bindings/python

# Static ffmpeg (one ~80 MB binary) instead of apt ffmpeg (~450 MB of codecs).
ARG TARGETARCH
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
        amd64) FF=amd64 ;; \
        arm64) FF=arm64 ;; \
        arm)   FF=armhf ;; \
        *)     FF=amd64 ;; \
    esac; \
    curl -fsSL "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${FF}-static.tar.xz" \
        -o /tmp/ff.tar.xz; \
    mkdir -p /tmp/ff; tar -xf /tmp/ff.tar.xz -C /tmp/ff --strip-components=1; \
    install -m 0755 /tmp/ff/ffmpeg /usr/local/bin/ffmpeg; \
    rm -rf /tmp/ff /tmp/ff.tar.xz

# Server sources come last so editing them does not bust the wheel cache.
COPY server/ server/

# Install into a self-contained venv, then strip the fat:
#  - scikit-build ships libvietasr.so x3 names x2 dirs (~400 MB) + headers;
#    keep just the one file cffi opens + the onnxruntime SONAME.
#  - drop pip/setuptools, bytecode and numpy tests.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir /wheels/*.whl ./server \
    && NV=/opt/venv/lib/python3.12/site-packages/vietasr/_native \
    && rm -rf "$NV/lib" "$NV/include" \
    && rm -f "$NV/libvietasr.so.0" "$NV/libvietasr.so.0.1.0" \
             "$NV/libonnxruntime.so" "$NV/libonnxruntime.so.1.20.1" \
    && SP=/opt/venv/lib/python3.12/site-packages \
    && rm -rf "$SP"/pip "$SP"/pip-* "$SP"/setuptools "$SP"/setuptools-* \
              "$SP"/pkg_resources "$SP"/_distutils_hack "$SP"/wheel "$SP"/wheel-* \
              "$SP"/numpy/tests "$SP"/numpy/_core/tests \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete

# ---- runtime --------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Drop the base image's pip/ensurepip — not needed to run the server.
RUN rm -rf /usr/local/lib/python3.12/site-packages/pip* \
           /usr/local/lib/python3.12/ensurepip \
    && find /usr/local/lib/python3.12 -name '__pycache__' -type d -prune \
            -exec rm -rf {} + 2>/dev/null || true

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg

ENV PATH=/opt/venv/bin:$PATH

RUN useradd --create-home --uid 10001 vietasr
USER vietasr
EXPOSE 8000

CMD ["vietasr-server", "--host", "0.0.0.0", "--port", "8000"]
