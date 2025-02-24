# Stage 1: Builder Stage
# Use an image that has both TensorRT and Triton support.
FROM nvcr.io/nvidia/tritonserver:24.05-py3 AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git wget git-lfs build-essential cmake curl python3-dev ninja-build && \
    git lfs install && \
    pip install --upgrade pip huggingface_hub

##########################
# Build Model Engine (TensorRT-LLM Engine)
##########################
# Clone TensorRT-LLM repository and install its requirements
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git /TensorRT-LLM
WORKDIR /TensorRT-LLM
RUN pip install -r requirements.txt

# # Download Llama 3.2 11B Vision model (adjust model name/path as needed)
# RUN mkdir -p /models/Llama-3.2-11B-Vision && \
#     huggingface-cli download meta-llama/Llama-3.2-11B-Vision --local-dir /models/Llama-3.2-11B-Vision

# # Build the visual engine with INT8 precision
# WORKDIR /TensorRT-LLM/examples/multimodal
# RUN python build_visual_engine.py \
#     --model_type mllama \
#     --model_path /models/Llama-3.2-11B-Vision \
#     --output_dir /model_engine \
#     --dtype int8

##########################
# Build Triton Backend
##########################
# Clone the Triton backend repository for TensorRT-LLM
WORKDIR /app
RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend.git /tensorrtllm_backend --recursive
WORKDIR /tensorrtllm_backend

# Install backend Python dependencies
RUN pip install -r requirements.txt

# Patch and build the backend using build.sh. Here we disable Docker-related commands to perform a local build.
# Patch and build the backend using build.sh by commenting out docker commands only
RUN chmod +x build.sh && \
    sed -i '/^docker run/ s/^/#/' build.sh && \
    sed -i '/^docker build/ s/^/#/' build.sh && \
    sed -i '/^docker pull/ s/^/#/' build.sh && \
    sed -i '/^docker push/ s/^/#/' build.sh && \
    ./build.sh --enable-gpu --build-type=Release --no-container-build \
      --cmake-args="-DTRITON_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release"


# At this point the backend build outputs are in /opt/tritonserver/backends/tensorrtllm_backend

# End of builder stage

# Stage 2: Runtime Stage
FROM nvcr.io/nvidia/tritonserver:24.05-py3 AS runtime
WORKDIR /app

# Copy the built model engine from builder
COPY --from=builder /model_engine/ /model_engine

# Copy the built Triton backend from builder
COPY --from=builder /opt/tritonserver/backends/tensorrtllm_backend /opt/tritonserver/backends/tensorrtllm_backend

# Copy your model repository (assumed to be in the local context as "model_repository")
COPY model_repository/ /opt/tritonserver/models/

# Expose the standard Triton ports
EXPOSE 8000 8001 8002

# Start Triton Inference Server with the provided model repository and backend config
CMD ["tritonserver", "--model-store=/opt/tritonserver/models", "--backend-config=tensorrtllm_backend,config.pb"]
