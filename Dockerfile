# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.0
ARG DEBIAN_FRONTEND=noninteractive
ARG BUNDLE_FLASH_ATTENTION=false

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS flashattn-builder

ARG DEBIAN_FRONTEND
ARG BUNDLE_FLASH_ATTENTION

ENV MAX_JOBS=32 \
    NVCC_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3 \
        python3-dev \
        python3-pip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U pip setuptools wheel

RUN mkdir -p /wheels

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
      python3 -m pip wheel --wheel-dir /wheels --no-build-isolation \
        git+https://github.com/Dao-AILab/flash-attention.git; \
    fi

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND
ARG BUNDLE_FLASH_ATTENTION

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U \
      "qwen-asr[vllm]" \
      "modelscope>=1.26.0" \
      "huggingface_hub[cli]>=0.30.0"

COPY --from=flashattn-builder /wheels /tmp/flashattn-wheels

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
      if ! ls /tmp/flashattn-wheels/*.whl >/dev/null 2>&1; then \
        echo "flash-attn wheel not found in builder output" >&2; \
        exit 1; \
      fi; \
      python3 -m pip install -U /tmp/flashattn-wheels/*.whl; \
    fi \
    && rm -rf /tmp/flashattn-wheels

COPY pyproject.toml README.md ./
COPY asr_service ./asr_service
COPY scripts ./scripts

RUN chmod +x /app/scripts/start.sh

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install . --no-deps

EXPOSE 8000

CMD ["./scripts/start.sh"]
