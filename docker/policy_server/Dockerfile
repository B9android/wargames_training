# Policy inference server Docker image (Epic E5.5).
#
# Serves exported ONNX or TorchScript policies via a lightweight Flask
# REST API.  Supports both runtimes in the same image; the active runtime
# is selected via the POLICY_FORMAT environment variable (or auto-detected
# from the model file extension).
#
# Build:
#   docker build -t policy_server docker/policy_server/
#
# Run (ONNX):
#   docker run -e POLICY_PATH=/models/actor.onnx \
#              -v $(pwd)/exports:/models \
#              -p 8080:8080 \
#              policy_server
#
# Run (TorchScript):
#   docker run -e POLICY_PATH=/models/actor.torchscript.pt \
#              -e POLICY_FORMAT=torchscript \
#              -v $(pwd)/exports:/models \
#              -p 8080:8080 \
#              policy_server
#
# Health check:
#   curl http://localhost:8080/health
#
# Predict:
#   curl -s -X POST http://localhost:8080/predict \
#        -H "Content-Type: application/json" \
#        -d '{"obs": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}'

FROM python:3.11-slim

LABEL maintainer="wargames_training" \
      description="Policy inference server for exported ONNX/TorchScript policies" \
      version="1.0"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Install CPU-only PyTorch (saves ~1 GB vs CUDA build) plus ONNX Runtime
# and Flask.
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    onnxruntime==1.18.0 \
    flask==3.0.3 \
    numpy==1.26.4

# ── Application ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY server.py /app/server.py

# Create a directory for model files (mount your exports here at runtime).
RUN mkdir -p /models

# ── Runtime configuration ─────────────────────────────────────────────────────
# POLICY_PATH must be overridden at runtime via -e POLICY_PATH=/models/...
ENV POLICY_PATH=""
ENV POLICY_FORMAT=""
ENV HOST="0.0.0.0"
ENV PORT="8080"

EXPOSE 8080

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "server.py"]
