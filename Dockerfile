# Production Dockerfile for MM Imaging Radiomics Pipeline
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-distutils \
    build-essential \
    git \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements-dev.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-distutils \
    libopenblas0 \
    libgomp1 \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 \
    TORCH_HOME=/workspace/.torch

# Create application directory
WORKDIR /workspace

# Copy source code
COPY src/ /workspace/src/
COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/
COPY README.md /workspace/

# Create necessary directories
RUN mkdir -p /workspace/{data,results,logs,.cache,.torch}

# Add git metadata
LABEL maintainer="PhD Researcher 6"
LABEL description="MM Imaging Pathology & Radiomics Pipeline"
LABEL version="0.1.0"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || exit 1

# Default entrypoint
ENTRYPOINT ["python", "-m", "src.orchestration"]

# Allow overriding entrypoint
CMD ["--help"]
