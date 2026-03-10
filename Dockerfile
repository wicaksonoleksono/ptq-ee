# =============================================================================
# PTQ × LayerSkip Benchmark — Dockerfile (GPU)
# =============================================================================
# Based on LayerSkip's Dockerfile (Miniconda + conda env pattern),
# upgraded to nvidia/cuda base for GPU inference + PTQ libraries.
#
# Build (run from PTQ/ root):
#   docker build -f scripts/Dockerfile -t ptq-layerskip:latest .
#
# Run examples:
#   # Download datasets
#   docker run --gpus all --rm \
#       -e HUGGINGFACE_TOKEN=your_token \
#       -v $(pwd)/model_cache:/root/.cache/huggingface \
#       ptq-layerskip:latest \
#       python scripts/00_download_data.py
#
#   # Quantize a model
#   docker run --gpus all --rm \
#       -e HUGGINGFACE_TOKEN=your_token \
#       -v $(pwd)/quantized_models:/app/quantized_models \
#       -v $(pwd)/model_cache:/root/.cache/huggingface \
#       ptq-layerskip:latest \
#       python scripts/01_quantize.py --model facebook/layerskip-llama2-7B --method awq
#
#   # Run benchmark
#   docker run --gpus all --rm \
#       -e HUGGINGFACE_TOKEN=your_token \
#       -v $(pwd)/logs:/app/logs \
#       -v $(pwd)/quantized_models:/app/quantized_models \
#       -v $(pwd)/model_cache:/root/.cache/huggingface \
#       ptq-layerskip:latest \
#       python scripts/02_run_benchmark.py \
#           --model facebook/layerskip-llama2-7B \
#           --ptq_method awq \
#           --task cnn_dm_summarization \
#           --num_samples 50
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (same as LayerSkip + build-essential for bitsandbytes)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (same as LayerSkip)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean --all --yes

ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda environment with Python 3.10 (same as LayerSkip)
RUN conda create --name ptq_layerskip python=3.10 -y

RUN echo "source activate ptq_layerskip" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=ptq_layerskip
ENV PATH=$CONDA_DIR/envs/ptq_layerskip/bin:$PATH

# Install PyTorch 2.2.1 with CUDA 12.1 via conda
# (GPU version — LayerSkip's Dockerfile uses cpuonly, we need GPU for PTQ)
RUN conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Upgrade pip
RUN pip install --upgrade pip

# Install GPTQ with its dedicated CUDA wheel first (before requirements.txt)
RUN pip install auto-gptq \
    --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/ \
    --no-cache-dir

# Copy requirements and install (cached layer — only rebuilds if requirements.txt changes)
COPY scripts/requirements.txt /app/scripts/requirements.txt
RUN pip install --no-cache-dir -r /app/scripts/requirements.txt

# Copy the full project
# Expected structure in /app:
#   /app/LayerSkip/     ← LayerSkip repo
#   /app/scripts/       ← our benchmark + PTQ scripts
COPY . /app

WORKDIR /app

# Pre-create all output directories so they exist as mount points.
# Mount these to the host with -v flags (see docker_run.sh) so results
# are NOT lost when the container exits.
RUN mkdir -p \
    /app/logs \
    /app/results \
    /app/figures \
    /app/quantized_models \
    /app/model_cache

# Declare as volumes — Docker will keep these writable and separable from
# the image layer. Always override with explicit -v mounts at runtime.
VOLUME ["/app/logs", "/app/results", "/app/figures", "/app/quantized_models", "/app/model_cache"]

# Set LayerSkip path for benchmark runner
ENV LAYERSKIP_DIR=/app/LayerSkip

# Entrypoint
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
