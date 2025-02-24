# Use the official Triton Server image with TensorRT-LLM support
FROM nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3

# Set working directory
WORKDIR /app

# Install additional dependencies (e.g., for Hugging Face model download)
RUN pip install --no-cache-dir \
    huggingface_hub \
    transformers>=4.43.0 \
    sentencepiece \
    protobuf

# Environment variables for Hugging Face (optional: for faster downloads)
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ARG HF_TOKEN
# Download LLaMA 3.2 3B model from Hugging Face
# Replace <YOUR_HF_TOKEN> with your Hugging Face token if required
RUN huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential && \
    huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir /app/model

# Convert the model to TensorRT-LLM checkpoint format
RUN python3 /opt/tritonserver/tensorrt_llm/examples/llama/convert_checkpoint.py \
    --model_dir /app/model \
    --output_dir /app/tllm_checkpoint \
    --dtype float16

# Build the TensorRT engine (optimized for single GPU, FP16 precision)
RUN trtllm-build \
    --checkpoint_dir /app/tllm_checkpoint \
    --output_dir /app/trt_engines \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_input_len 2048 \
    --max_output_len 512

# Clone the TensorRT-LLM backend repository for Triton configuration
RUN git clone --branch v0.10.0 https://github.com/triton-inference-server/tensorrtllm_backend.git /app/tensorrtllm_backend

# Set up Triton model repository
RUN mkdir -p /app/model_repository/inflight_batcher_llm/tensorrt_llm/1 && \
    cp -r /app/trt_engines/* /app/model_repository/inflight_batcher_llm/tensorrt_llm/1/ && \
    cp -r /app/tensorrtllm_backend/all_models/inflight_batcher_llm/* /app/model_repository/inflight_batcher_llm/ && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py -i /app/model_repository/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
        engine_dir:/app/model_repository/inflight_batcher_llm/tensorrt_llm/1/,max_tokens_in_paged_kv_cache:5120 && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py -i /app/model_repository/inflight_batcher_llm/preprocessing/config.pbtxt \
        tokenizer_dir:/app/model,tokenizer_type:llama && \
    python3 /app/tensorrtllm_backend/tools/fill_template.py -i /app/model_repository/inflight_batcher_llm/postprocessing/config.pbtxt \
        tokenizer_dir:/app/model,tokenizer_type:llama

# Expose Triton HTTP and gRPC ports
EXPOSE 8000 8001

# Command to start Triton Server
CMD ["tritonserver", "--model-repository=/app/model_repository/inflight_batcher_llm"]