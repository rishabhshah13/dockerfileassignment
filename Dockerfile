# Stage 1: Builder Stage
FROM nvcr.io/nvidia/tritonserver:24.05-py3 AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget git-lfs build-essential cmake curl python3-dev ninja-build && \
    git lfs install && \
    pip3 install --upgrade pip huggingface_hub && \
    rm -rf /var/lib/apt/lists/*

# Clone TensorRT-LLM repository
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git /TensorRT-LLM
WORKDIR /TensorRT-LLM

# Enable Universe repository and install libopenmpi-dev, then install tensorrt_llm from NVIDIA's PyPI
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends libopenmpi-dev && \
    pip3 install --upgrade pip setuptools && \
    pip3 install tensorrt_llm --extra-index-url https://pypi.nvidia.com

# Download LLaMA 3.2 11B Vision model using huggingface-cli
# Note: Requires authentication if the model is gated (set HF_TOKEN env var if needed)
# ARG HF_TOKEN
# RUN mkdir -p /models/Llama-3.2-11B-Vision && \
#     huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
#     --local-dir /models/Llama-3.2-11B-Vision \
#     --token ${HF_TOKEN}

# # Build the TensorRT-LLM engine with INT8 precision
# WORKDIR /TensorRT-LLM/examples/multimodal
# RUN python3 build_visual_engine.py \
#     --model_type mllama \
#     --model_path /models/Llama-3.2-11B-Vision \
#     --output_dir /model_engine \
#     --dtype int8


ARG HF_TOKEN
RUN mkdir -p /models/Llama-3.2-3B && \
    huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --local-dir /models/Llama-3.2-3B \
    --token ${HF_TOKEN}

# Build the TensorRT-LLM engine with INT8 precision
WORKDIR /TensorRT-LLM/examples/multimodal
RUN python3 build_visual_engine.py \
    --model_type mllama \
    --model_path /models/Llama-3.2-3B \
    --output_dir /model_engine \
    --dtype int8

# Stage 2: Runtime Stage
FROM nvcr.io/nvidia/tritonserver:24.05-py3 AS runtime
WORKDIR /app

# Install runtime dependencies (minimal set for Triton + TensorRT-LLM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip && \
    pip3 install --upgrade pip numpy grpcio-tools && \
    rm -rf /var/lib/apt/lists/*

# Copy the built TensorRT-LLM engine from the builder stage
COPY --from=builder /model_engine/ /model_engine/

# Copy the TensorRT-LLM backend from the builder stage
# Note: The backend may already be in the base image; copying ensures consistency
COPY --from=builder /opt/tritonserver/backends/tensorrtllm_backend /opt/tritonserver/backends/tensorrtllm_backend

# Copy the model repository from the local context
# This should contain your model config (e.g., config.pbtxt) pointing to /model_engine
COPY ./model_repository/ /opt/tritonserver/models/

# Set environment variables for Triton
ENV TRITON_SERVER_PATH=/opt/tritonserver/bin/tritonserver
ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH

# Expose default Triton ports: HTTP (8000), gRPC (8001), Metrics (8002)
EXPOSE 8000 8001 8002

# Start Triton Inference Server with the model repository and backend config
CMD ["tritonserver", "--model-store=/opt/tritonserver/models", "--backend-config=tensorrtllm,verbose=true"]
