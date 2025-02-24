FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 AS runtime
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# COPY scripts/requirements.txt .
# RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Clone TensorRT-LLM for scripts
RUN git clone --branch v0.17.0 https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm && \
    cd /app/tensorrt_llm && git submodule update --init --recursive

# Copy scripts
COPY scripts/build_engines.py .
COPY scripts/entrypoint.sh .

# Copy Triton model repo
COPY model_repo/multimodal_ifb /models/multimodal_ifb/

# Fill Triton configs
RUN python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/multimodal_ifb/tensorrt_llm/config.pbtxt \
    triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:/models/tensorrt_llm/1/,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:TYPE_BF16,logits_datatype:TYPE_FP32 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/multimodal_ifb/preprocessing/config.pbtxt \
    tokenizer_dir:/models/Llama-3.2-11B-Vision,triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:/models/multimodal_encoders/1/,engine_dir:/models/tensorrt_llm/1/,max_num_images:1 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/multimodal_ifb/postprocessing/config.pbtxt \
    tokenizer_dir:/models/Llama-3.2-11B-Vision,triton_max_batch_size:8,postprocessing_instance_count:1 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/multimodal_ifb/ensemble/config.pbtxt \
    triton_max_batch_size:8,logits_datatype:TYPE_FP32 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/multimodal_ifb/tensorrt_llm_bls/config.pbtxt \
    triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32 && \
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
    -i /models/multimodal_ifb/multimodal_encoders/config.pbtxt \
    triton_max_batch_size:8,visual_model_path:/models/multimodal_encoders/1/,encoder_input_features_data_type:TYPE_BF16,hf_model_path:/models/Llama-3.2-11B-Vision

# Set HF_TOKEN as env var for runtime
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Set entrypoint
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

EXPOSE 8000 8001 8002