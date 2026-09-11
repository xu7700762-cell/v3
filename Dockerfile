FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONHASHSEED=2001 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vestibular-fusion
COPY . .

RUN python3.10 -m pip install --upgrade pip packaging \
    && python3.10 -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0+cu128 \
    && python3.10 -m pip install --no-build-isolation mamba-ssm==2.3.1 \
    && python3.10 -m pip install -e .

ENTRYPOINT ["python3.10"]
CMD ["scripts/verify_reproduction.py"]
