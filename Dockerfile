# FROM nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3
FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

WORKDIR /app

# Install dependencies with pinned versions to avoid incompatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip && \
    # pip3 install --upgrade pip && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install huggingface_hub transformers torch timm grpcio-tools \
                 numpy==1.26.4 pandas==1.5.3 datasets==2.14.0 && \
    rm -rf /var/lib/apt/lists/*

# Download LLaMA 3.2 11B Vision Instruct model
ARG HF_TOKEN
RUN mkdir -p /models/Llama-3.2-11B-Vision && \
    huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
    --local-dir /models/Llama-3.2-11B-Vision \
    --token ${HF_TOKEN}

# Clone TensorRT-LLM repo for build_visual_engine.py
# RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm
RUN git clone --branch v0.17.0 https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm

# Clone tensorrtllm_backend for Triton templates
# RUN git clone --branch v0.10.0 https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend
RUN git clone --branch triton-llm/v0.17.0 https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend

# Set up Triton model repository (corrected structure)
RUN mkdir -p /model_engine && \
    mkdir -p /opt/tritonserver/models/tensorrt_llm/1 && \
    cp -r /app/tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/* /opt/tritonserver/models/tensorrt_llm/ && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py -i /opt/tritonserver/models/tensorrt_llm/config.pbtxt \
        engine_dir:/opt/tritonserver/models/tensorrt_llm/1/,max_tokens_in_paged_kv_cache:5120,tp_size:2 && \
    sed -i '/input \[/a\  {\n    name: "pixel_values"\n    data_type: TYPE_FP32\n    dims: [ 1, 3, 224, 224 ]\n  },' \
        /opt/tritonserver/models/tensorrt_llm/config.pbtxt

# Create entrypoint script with error checking
RUN cat <<'EOF' > /app/entrypoint.sh
#!/bin/bash
set -e  # Exit on error
if [ ! -f /model_engine/completed ]; then
  echo "Building TensorRT engine with 2 GPUs..."
  cd /app/tensorrt_llm/examples/multimodal
  python3 build_visual_engine.py \
    --model_type mllama \
    --model_path /models/Llama-3.2-11B-Vision \
    --output_dir /model_engine \
    --dtype int8 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --tp_size 2
  if [ $? -ne 0 ]; then
    echo "Engine build failed!"
    exit 1
  fi
  cp -r /model_engine/* /opt/tritonserver/models/tensorrt_llm/1/
  touch /model_engine/completed
  echo "Engine build complete."
fi
echo "Starting Triton Server with 2 GPUs..."
tritonserver --model-store=/opt/tritonserver/models --backend-config=tensorrtllm,verbose=true
EOF
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8001 8002
ENTRYPOINT ["/app/entrypoint.sh"]