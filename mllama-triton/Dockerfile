FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 AS runtime
WORKDIR /app

# Install minimal system dependencies and required libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libopenmpi-dev && \
    apt-get purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    rm -rf /var/lib/apt/lists/*

# Install bitsandbytes for Unsloth 4-bit and compressed_tensors for Neural Magic FP8
RUN pip3 install bitsandbytes==0.43.1
RUN pip3 install compressed-tensors

# Clone TensorRT-LLM and tensorrtllm_backend
RUN git clone --branch v0.17.0 https://github.com/NVIDIA/TensorRT-LLM.git /app/tensorrt_llm && \
    cd /app/tensorrt_llm && git submodule update --init --recursive
RUN git clone --branch main https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend && \
    cd /app/tensorrtllm_backend && git submodule update --init --recursive

# Copy scripts
COPY scripts/build_engines.py .
COPY scripts/entrypoint.sh .

# Copy Triton model repo
COPY model_repo/multimodal_ifb /models/multimodal_ifb/

# Fill Triton configs dynamically based on model (default to meta-llama for now)
RUN python3 /app/tensorrtllm_backend/tools/fill_template.py \
    -i /models/multimodal_ifb/tensorrt_llm/config.pbtxt \
    triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:/models/tensorrt_llm/1/,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:TYPE_INT4,logits_datatype:TYPE_FP32 && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py \
    -i /models/multimodal_ifb/preprocessing/config.pbtxt \
    tokenizer_dir:/models/meta-llama-Llama-3.2-11B-Vision-Instruct,triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:/models/multimodal_encoders/1/,engine_dir:/models/tensorrt_llm/1/,max_num_images:1 && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py \
    -i /models/multimodal_ifb/postprocessing/config.pbtxt \
    tokenizer_dir:/models/meta-llama-Llama-3.2-11B-Vision-Instruct,triton_max_batch_size:8,postprocessing_instance_count:1 && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py \
    -i /models/multimodal_ifb/ensemble/config.pbtxt \
    triton_max_batch_size:8,logits_datatype:TYPE_FP32 && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py \
    -i /models/multimodal_ifb/tensorrt_llm_bls/config.pbtxt \
    triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32 && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py \
    -i /models/multimodal_ifb/multimodal_encoders/config.pbtxt \
    triton_max_batch_size:8,visual_model_path:/models/multimodal_encoders/1/,encoder_input_features_data_type:TYPE_INT4,hf_model_path:/models/meta-llama-Llama-3.2-11B-Vision-Instruct

# Set HF_TOKEN as env var for runtime
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Set entrypoint
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

EXPOSE 8000 8001 8002