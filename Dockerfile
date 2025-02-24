# Stage 1: Builder Stage (Using Triton Server with TensorRT-LLM Support)
FROM nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 AS builder
WORKDIR /app

# Install dependencies for model download and mllama engine building
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip && \
    pip3 install --upgrade pip huggingface_hub transformers torch timm && \
    rm -rf /var/lib/apt/lists/*

# Download LLaMA 3.2 11B Vision Instruct model
ARG HF_TOKEN
RUN mkdir -p /models/Llama-3.2-11B-Vision && \
    huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
    --local-dir /models/Llama-3.2-11B-Vision \
    --token ${HF_TOKEN}

# Build the TensorRT-LLM engine for mllama with build_visual_engine.py
WORKDIR /opt/tritonserver/tensorrt_llm/examples/multimodal
RUN python3 build_visual_engine.py \
    --model_type mllama \
    --model_path /models/Llama-3.2-11B-Vision \
    --output_dir /model_engine \
    --dtype int8 \
    --max_input_len 2048 \
    --max_output_len 512

# Clone the official tensorrtllm_backend for Triton templates
RUN git clone --branch v0.10.0 https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend

# Stage 2: Runtime Stage (Using Same Triton Server Base)
FROM nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 AS runtime
WORKDIR /app

# Install minimal runtime dependencies for Triton
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip && \
    pip3 install --upgrade pip numpy grpcio-tools && \
    rm -rf /var/lib/apt/lists/*

# Copy the built TensorRT-LLM engine from the builder stage
COPY --from=builder /model_engine/ /opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/1/

# Copy the tensorrtllm_backend templates and populate the model repository
COPY --from=builder /app/tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/models/inflight_batcher_llm/
RUN python3 /opt/tritonserver/tensorrt_llm/tools/fill_template.py -i /opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
        engine_dir:/opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/1/,max_tokens_in_paged_kv_cache:5120

# Adjust config.pbtxt to include image input for mllama
RUN sed -i '/input \[/a\  {\n    name: "pixel_values"\n    data_type: TYPE_FP32\n    dims: [ 1, 3, 224, 224 ]\n  },' \
    /opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/config.pbtxt

# Expose Triton ports: HTTP (8000), gRPC (8001), Metrics (8002)
EXPOSE 8000 8001 8002

# Start Triton Inferenc