# Stage 1: Build Stage - Model Engine Creation
FROM nvcr.io/nvidia/tensorrt:23.10-py3 AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y git wget
RUN pip install --upgrade pip
RUN pip install huggingface_hub

# Clone TensorRT-LLM
RUN git clone https://github.com/NVIDIA/TensorRT-LLM /TensorRT-LLM
WORKDIR /TensorRT-LLM
RUN pip install -r requirements.txt

# Download Llama 3.2 11B Vision model
RUN mkdir -p /models/Llama-3.2-11B-Vision
RUN huggingface-cli download meta-llama/Llama-3.2-11B-Vision --local-dir /models/Llama-3.2-11B-Vision

# Build the visual engine with INT8 precision
WORKDIR /TensorRT-LLM/examples/multimodal
RUN python build_visual_engine.py --model_type mllama \
    --model_path /models/Llama-3.2-11B-Vision \
    --output_dir /tmp/mllama/trt_engines/encoder/ \
    --dtype int8

# Stage 2: Triton Backend Build Stage
FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS triton-builder
WORKDIR /app

# Install Git LFS, build tools, and other dependencies
RUN apt-get update && apt-get install -y \
    git-lfs \
    build-essential \
    cmake \
    curl \
    python3-dev \
    ninja-build \
    && git lfs install

# Clone tensorrtllm_backend with submodules
RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend /tensorrtllm_backend --recursive
WORKDIR /tensorrtllm_backend

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Build the backend using build.sh with local build (no Docker-in-Docker)
# Thoroughly patch build.sh to remove all Docker dependencies and handle -f, -t, etc.
RUN chmod +x build.sh && \
    # Comment out all Docker-related commands
    sed -i 's/docker run/#docker run/g' build.sh && \
    sed -i 's/docker build/#docker build/g' build.sh && \
    sed -i 's/docker pull/#docker pull/g' build.sh && \
    sed -i 's/docker push/#docker push/g' build.sh && \
    # Remove or comment out standalone arguments like --build-arg, -f, -t, --rm, etc.
    sed -i 's/--build-arg/#--build-arg/g' build.sh && \
    sed -i 's/-f/#-f/g' build.sh && \
    sed -i 's/-t/#-t/g' build.sh && \
    sed -i 's/--rm/#--rm/g' build.sh && \
    # Ensure build.sh proceeds with a local build using correct arguments
    ./build.sh --enable-gpu --build-type=Release --no-container-build --cmake-args="-DTRITON_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release"

# If build.sh fails, try building with CMake in the inflight_batcher_llm directory (if applicable)
# RUN mkdir build && cd build && \
#     cmake ../inflight_batcher_llm -DTRITON_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release -DTRITON_BACKEND_PATH=/opt/tritonserver/backends && \
#     make -j$(nproc) install

# Stage 3: Runtime Stage
FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS runtime
WORKDIR /app

# Copy model engine from builder stage
COPY --from=builder /tmp/mllama/trt_engines/encoder/ /model_engine

# Copy backend from triton-builder stage
COPY --from=triton-builder /opt/tritonserver/backends/tensorrtllm_backend /opt/tritonserver/backends/tensorrtllm_backend

# Copy your model repository structure into Triton's model store
COPY model_repository/ /opt/tritonserver/models/

# Configure and start Triton server
CMD ["tritonserver", "--model-store=/opt/tritonserver/models", "--backend-config=tensorrtllm_backend,config.pb"]