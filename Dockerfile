ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BUNDLE_FLASH_ATTENTION=true

RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    libsndfile1 \
    ccache \
    software-properties-common \
    ffmpeg \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*
EOF

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN git lfs install

WORKDIR /app

ENV MAX_JOBS=32
ENV NVCC_THREADS=2
ENV CCACHE_DIR=/root/.cache/ccache
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -U pip setuptools wheel

RUN apt remove python3-blinker -y

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -U \
      "qwen-asr[vllm]" \
      "modelscope>=1.26.0" \
      "huggingface_hub[cli]>=0.30.0"

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
      pip3 install -U flash-attn --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git; \
    fi

COPY pyproject.toml ./pyproject.toml
COPY asr_service ./asr_service
COPY scripts ./scripts

RUN chmod +x /app/scripts/start.sh

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install .

EXPOSE 8000

CMD ["./scripts/start.sh"]
