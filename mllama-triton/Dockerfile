FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 AS runtime
WORKDIR /app

# Install minimal system dependencies and required libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libopenmpi-dev && \
    apt-get purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    rm -rf /var/lib/apt/lists/*

# Install bitsandbytes for Unsloth 4-bit and compressed_tensors for Neural Magic FP8
RUN pip3 install bitsandbytes==0.43.1
RUN pip3 install compressed-tensors

# Clone TensorRT-LLM and tensorrtllm_backend
RUN git clone --branch v0.17.0 https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm && \
    cd /app/tensorrt_llm && git submodule update --init --recursive
RUN git clone --branch main https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend && \
    cd /app/tensorrtllm_backend && git submodule update --init --recursive

# Copy scripts
COPY scripts/build_engines.py .
COPY scripts/entrypoint.sh .

# Set HF_TOKEN as env var for runtime
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Set entrypoint
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

EXPOSE 8000 8001 8002