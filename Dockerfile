# =============================================================================
# PTQ × LayerSkip Benchmark — Dockerfile (GPU)
# =============================================================================
# Build (from PTQ/ root):
#   docker build -f scripts/Dockerfile -t ptq-layerskip:latest .
#
# All runtime usage goes through the Makefile. Don't docker run manually.
#   make build
#   make download
#   make quantize-awq
#   make benchmark METHOD=awq STRATEGY=self_speculative
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Miniconda (same pattern as LayerSkip/Dockerfile)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean --all --yes
ENV PATH=$CONDA_DIR/bin:$PATH

# Conda env — Python 3.10, GPU PyTorch 2.2.1 + CUDA 12.1
RUN conda create --name ptq python=3.10 -y
RUN echo "source activate ptq" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=ptq
ENV PATH=$CONDA_DIR/envs/ptq/bin:$PATH

RUN conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    pytorch-cuda=12.1 -c pytorch -c nvidia -y

RUN pip install --upgrade pip

# GPTQ CUDA wheel first (avoids conflicts with generic pip install)
RUN pip install auto-gptq \
    --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/ \
    --no-cache-dir

# Python deps (cached layer — only rebuilds when requirements.txt changes)
COPY scripts/requirements.txt /app/scripts/requirements.txt
RUN pip install --no-cache-dir -r /app/scripts/requirements.txt

# Copy project
COPY . /app
WORKDIR /app

# Output dirs — mount these to host with -v so results survive container exit
RUN mkdir -p /app/logs /app/results /app/figures /app/quantized_models /app/model_cache
VOLUME ["/app/logs", "/app/results", "/app/figures", "/app/quantized_models", "/app/model_cache"]

ENV LAYERSKIP_DIR=/app/LayerSkip
