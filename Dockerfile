# Stage 1: Base image with CUDA requirements
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    wget \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build TensorRT-LLM
FROM base AS tensorrt_llm_builder

WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git && \
    cd TensorRT-LLM && \
    # Remove the problematic flashinfer-python line
    sed -i '/flashinfer-python/d' requirements.txt && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    # Install flashinfer-python separately using the direct URL with egg fragment
    pip3 install "flashinfer-python @ https://files.pythonhosted.org/packages/6c/e9/5d6adcf888922a17c6fc52a0e5bed78785239af1219f41e1073b063a07ff/flashinfer_python-0.2.0.post1.tar.gz#egg=flashinfer-python" && \
    python3 scripts/build_wheel.py --build_dir=build && \
    pip3 install build/tensorrt_llm*.whl

# Stage 3: Model conversion and optimization
FROM tensorrt_llm_builder AS model_builder

WORKDIR /workspace

# Install additional requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Download and convert model (you'll need to provide access token)
ENV HF_TOKEN="your_huggingface_token"
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('meta-llama/Llama-3-11b-vision', token='${HF_TOKEN}')"

# Convert model to TensorRT format
COPY convert_model.py .
RUN python3 convert_model.py \
    --model_dir Llama-3-11b-vision \
    --output_dir llama3_trt_checkpoint

# Build TensorRT engine
RUN trtllm-build \
    --checkpoint_dir llama3_trt_checkpoint \
    --output_dir llama3_engines \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16 \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --remove_input_padding enable \
    --context_fmha enable \
    --use_inflight_batching true

# Stage 4: Triton Server Runtime
FROM nvcr.io/nvidia/tritonserver:24.02-py3 AS runtime

WORKDIR /workspace

# Copy TensorRT engines and model repository
COPY --from=model_builder /workspace/llama3_engines /models/llama3_vision/1/
COPY model_repository /models/

# Copy necessary scripts and configurations
COPY triton_model_config/* /models/

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton Server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
