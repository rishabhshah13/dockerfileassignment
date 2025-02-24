# Single Stage: Triton Server with TensorRT-LLM Support
FROM nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3
WORKDIR /app

# Install dependencies for model download, engine building, and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip && \
    pip3 install --upgrade pip huggingface_hub transformers torch timm numpy grpcio-tools && \
    rm -rf /var/lib/apt/lists/*

# Download LLaMA 3.2 11B Vision Instruct model
ARG HF_TOKEN
RUN mkdir -p /models/Llama-3.2-11B-Vision && \
    huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
    --local-dir /models/Llama-3.2-11B-Vision \
    --token ${HF_TOKEN}

# Clone TensorRT-LLM repo for build_visual_engine.py
RUN git clone --branch v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm

# Clone tensorrtllm_backend for Triton templates and tools
RUN git clone --branch v0.10.0 https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend

# Create engine output directory and model repository
RUN mkdir -p /model_engine && \
    mkdir -p /opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/1 && \
    cp -r /app/tensorrtllm_backend/all_models/inflight_batcher_llm/* /opt/tritonserver/models/inflight_batcher_llm/ && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py -i /opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
        engine_dir:/opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/1/,max_tokens_in_paged_kv_cache:5120,tp_size:2 && \
    sed -i '/input \[/a\  {\n    name: "pixel_values"\n    data_type: TYPE_FP32\n    dims: [ 1, 3, 224, 224 ]\n  },' \
        /opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/config.pbtxt

# Create entrypoint script to build engine with multi-GPU and start Triton
RUN echo '#!/bin/bash\n\
if [ ! -f /model_engine/completed ]; then\n\
  echo "Building TensorRT engine with 2 GPUs..."\n\
  cd /app/tensorrt_llm/examples/multimodal && \\\n\
  python3 build_visual_engine.py \\\n\
    --model_type mllama \\\n\
    --model_path /models/Llama-3.2-11B-Vision \\\n\
    --output_dir /model_engine \\\n\
    --dtype int8 \\\n\
    --max_input_len 2048 \\\n\
    --max_output_len 512 \\\n\
    --tp_size 2 && \\\n\
  cp -r /model_engine/* /opt/tritonserver/models/inflight_batcher_llm/tensorrt_llm/1/ && \\\n\
  touch /model_engine/completed\n\
  echo "Engine build complete."\n\
fi\n\
echo "Starting Triton Server with 2 GPUs..."\n\
tritonserver --model-store=/opt/tritonserver/models --backend-config=tensorrtllm,verbose=true' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose Triton ports: HTTP (8000), gRPC (8001), Metrics (8002)
EXPOSE 8000 8001 8002

# Use entrypoint to build engine and start Triton
ENTRYPOINT ["/app/entrypoint.sh"]