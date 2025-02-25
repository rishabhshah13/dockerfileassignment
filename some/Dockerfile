# Build stage
FROM ubuntu:22.04 AS build

# Install dependencies
RUN apt-get update && apt-get install -y git

# Clone TensorRT-LLM
RUN git clone --branch v0.17.0 https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm && \
    cd /app/tensorrt_llm && git submodule update --init --recursive

# Clone tensorrtllm_backend
# RUN git clone --branch triton-llm/v0.17.0 https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend && \
#     cd /app/tensorrtllm_backend && git submodule update --init --recursive
RUN git clone --branch main https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend && \
    cd /app/tensorrtllm_backend && git submodule update --init --recursive

# Runtime stage
FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 AS runtime

# Copy cloned repositories to the runtime image
COPY --from=build /app /app

# Add an entrypoint script to handle model download, engine building, and server startup
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]