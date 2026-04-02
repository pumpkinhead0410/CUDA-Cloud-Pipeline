# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: CUDA / C++ build environment
#   - Compile the custom CUDA kernels into a shared library (libcuda_kernels.so)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cuda-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake=3.22.* \
        ninja-build \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY src/cuda/kernels/ ./kernels/

RUN cmake -S kernels -B kernels/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build kernels/build --parallel "$(nproc)" \
    && cmake --install kernels/build --prefix /install


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Python dependency builder
#   - Install Python wheels into an isolated prefix so the runtime image stays
#     lean.  Uses --no-cache-dir and --no-deps for reproducibility.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS python-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only the requirements first to leverage Docker layer cache
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Final runtime image
#   - Minimal CUDA runtime + Python venv + compiled shared lib
#   - Runs as a non-root user for security
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

LABEL org.opencontainers.image.title="CUDA Cloud Inference API"
LABEL org.opencontainers.image.description="TensorRT-backed ML inference API on AWS EKS"
LABEL org.opencontainers.image.source="https://github.com/pumpkinhead0410/CUDA-Cloud-Pipeline"

# Runtime OS packages (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from python-builder
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy compiled CUDA kernel shared library
COPY --from=cuda-builder /install/lib/libcuda_kernels.so /usr/local/lib/
RUN ldconfig

# Copy application source
WORKDIR /app
COPY src/ ./src/

# Non-root user
RUN groupadd --gid 10001 appgroup \
    && useradd  --uid 10001 --gid appgroup --no-create-home appuser \
    && chown -R appuser:appgroup /app
USER appuser

# Runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENGINE_PATH=/models/model.engine \
    MODEL_NAME=default \
    USE_STUB=0 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Use uvicorn with multiple workers for production
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]
