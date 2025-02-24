# Stage 1: Build Engines
FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 AS builder
# FROM nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 AS builder
# FROM nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3 AS builder

WORKDIR /app

Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libopenmpi-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*
# COPY scripts/requirements.txt .
# RUN pip3 install -r requirements.txt

# Clone TensorRT-LLM for scripts
RUN git clone --branch v0.17.0 https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm && \
    cd /app/tensorrt_llm && git submodule update --init --recursive

# Download LLaMA 3.2 11B Vision
ARG HF_TOKEN
RUN mkdir -p /models/Llama-3.2-11B-Vision && \
    huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
    --local-dir /models/Llama-3.2-11B-Vision --token ${HF_TOKEN}

# Build engines
COPY scripts/build_engines.py .
RUN chmod +x build_engines.py && \
    python3 build_engines.py --model_path /models/Llama-3.2-11B-Vision --output_dir /model_engine

# Stage 2: Runtime with Triton Server
FROM nvcr.io/nvidia/tritonserver:25.01-py3 AS runtime
WORKDIR /opt/tritonserver

# Copy engines
COPY --from=builder /model_engine/llm /models/tensorrt_llm/1/
COPY --from=builder /model_engine/vision /models/multimodal_encoders/1/

# Copy and configure Triton model repo
COPY model_repo/multimodal_ifb /models/
RUN python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/tensorrt_llm/config.pbtxt \
    triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:/models/tensorrt_llm/1/,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:TYPE_BF16,logits_datatype:TYPE_FP32 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/preprocessing/config.pbtxt \
    tokenizer_dir:/models/Llama-3.2-11B-Vision,triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:/models/multimodal_encoders/1/,engine_dir:/models/tensorrt_llm/1/,max_num_images:1 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/postprocessing/config.pbtxt \
    tokenizer_dir:/models/Llama-3.2-11B-Vision,triton_max_batch_size:8,postprocessing_instance_count:1 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/ensemble/config.pbtxt \
    triton_max_batch_size:8,logits_datatype:TYPE_FP32 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/multimodal_encoders/config.pbtxt \
    triton_max_batch_size:8,visual_model_path:/models/multimodal_encoders/1/,encoder_input_features_data_type:TYPE_BF16,hf_model_path:/models/Llama-3.2-11B-Vision

# Copy HF model for tokenizer (optional, if preprocessing needs it)
COPY --from=builder /models/Llama-3.2-11B-Vision /models/Llama-3.2-11B-Vision

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton Server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]