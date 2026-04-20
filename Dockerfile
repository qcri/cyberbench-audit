# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip

WORKDIR /app




# Copy only non-ignored files (handled by .dockerignore)
COPY . .


# Install Python dependencies (including vllm and torch with CUDA)
RUN python3 -m pip install --no-cache-dir -r requirements.txt


# Set default command
CMD ["bash"]
