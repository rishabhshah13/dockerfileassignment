# Build stage
FROM ubuntu:22.04 AS build

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip
RUN pip install jupyter

# Clone TensorRT-LLM with matching version (adjust if needed)
RUN git clone --branch v0.17.0 https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm && \
    cd /app/tensorrt_llm && git submodule update --init --recursive
# RUN git clone --branch rel https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm && \
#     cd /app/tensorrt_llm && git submodule update --init --recursive

RUN git clone --branch triton-llm/v0.17.0 https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend && \
    cd /app/tensorrtllm_backend && git submodule update --init --recursive
# RUN git clone --branch rel https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend && \
#     cd /app/tensorrtllm_backend && git submodule update --init --recursive


# Clone tensorrtllm_backend with a specific tag (replace with actual tag)
# RUN git clone --branch rel-25.01 https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend && \
#     cd /app/tensorrtllm_backend && git submodule update --init --recursive

# Runtime stage
FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 AS runtime

# Install Jupyter (Optional)
RUN pip install jupyter

# Set NVIDIA runtime for GPU access
ENV NVIDIA_VISIBLE_DEVICES=0,1  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy built TensorRT-LLM and entrypoint script
COPY --from=build /app /app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]