#!/bin/bash

# Define directories for persistent storage
MODEL_DIR=/model_data/hf_models/Llama-3.2-11B-Vision
CKPT_DIR=/model_data/trt_ckpts/Llama-3.2-11B-Vision
LLM_ENGINE_DIR=/model_data/trt_engines/Llama-3.2-11B-Vision/llm/bf16/2-gpu
VISION_ENGINE_DIR=/model_data/trt_engines/Llama-3.2-11B-Vision/vision/bf16/1-gpu
MODEL_REPO_DIR=/model_data/multimodal_ifb


# Add these checks to your entrypoint.sh
echo "Checking repository structure..."
ls -la /app/tensorrt_llm
ls -la /app/tensorrt_llm/examples

# Login to Hugging Face if token is provided
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN
else
    echo "HF_TOKEN environment variable not set. Please provide your Hugging Face token."
    exit 1
fi

# Function to download model with retry
download_model() {
    local max_retries=3
    local retry_count=0
    while [ $retry_count -lt $max_retries ]; do
        huggingface-cli download meta-llama/Llama-3.2-11B-Vision --local-dir $MODEL_DIR
        if [ $? -eq 0 ]; then
            echo "Model downloaded successfully."
            return 0
        else
            echo "Download failed. Retrying ($((retry_count + 1))/$max_retries)..."
            rm -rf $MODEL_DIR  # Remove incomplete download
            retry_count=$((retry_count + 1))
            sleep 10
        fi
    done
    echo "Failed to download model after $max_retries attempts."
    exit 1
}

# Download model if not already present
if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/original/consolidated.pth" ]; then
    echo "Downloading model..."
    download_model
fi



# Build engines if not already built
if [ ! -d "$LLM_ENGINE_DIR" ] || [ ! -d "$VISION_ENGINE_DIR" ]; then
    echo "Building engines..."
    # Convert checkpoint for decoder
    python /app/tensorrt_llm/examples/mllama/convert_checkpoint.py  \
        --model_dir $MODEL_DIR \
        --output_dir $CKPT_DIR \
        --dtype bfloat16
    # Build decoder engine with tensor parallelism for 2 GPUs
    trtllm-build \
        --checkpoint_dir $CKPT_DIR \
        --output_dir $LLM_ENGINE_DIR \
        --max_num_tokens 4096 \
        --max_seq_len 2048 \
        --workers 1 \
        --gemm_plugin blfoat16 \
        --max_batch_size 4 \
        --max_encoder_input_len 6404 \
       # --tp_size 2  # Use both GPUs
    # Build vision encoder engine (single GPU)
    python /app/tensorrt_llm/examples/multimodal/build_visual_engine.py \
        --model_type mllama \
        --model_path $MODEL_DIR \
        --output_dir $VISION_ENGINE_DIR \
        --max_batch_size 8
fi


ls -la $MODEL_REPO_DIR
echo "outside Setting up model repository..."
 
# Set up model repository if not already present
if [  -d "$MODEL_REPO_DIR" ]; then
    echo "Setting up model repository..."
    # Copy multimodal model repository (verify this path exists in your tensorrtllm_backend)
    cp -r /app/tensorrtllm_backend/all_models/multimodal/* $MODEL_REPO_DIR
    cp -r //app/tensorrtllm_backend/all_models/inflight_batcher_llm/*  $MODEL_REPO_DIR

    

    ls -la $MODEL_REPO_DIR
    # Fill templates for config.pbtxt files
    python /app/tensorrtllm_backend/tools/fill_template.py -i $MODEL_REPO_DIR/preprocessing/config.pbtxt \
        tokenizer_dir:$MODEL_DIR,triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:$VISION_ENGINE_DIR,engine_dir:$LLM_ENGINE_DIR,max_num_images:1,max_queue_delay_microseconds:20000
    python /app/tensorrtllm_backend/tools/fill_template.py -i $MODEL_REPO_DIR/postprocessing/config.pbtxt \
        tokenizer_dir:$MODEL_DIR,triton_max_batch_size:8,postprocessing_instance_count:1
    python /app/tensorrtllm_backend/tools/fill_template.py -i $MODEL_REPO_DIR/ensemble/config.pbtxt \
        triton_max_batch_size:8,logits_datatype:TYPE_FP32
    python /app/tensorrtllm_backend/tools/fill_template.py -i $MODEL_REPO_DIR/tensorrt_llm_bls/config.pbtxt \
        triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32
    python /app/tensorrtllm_backend/tools/fill_template.py -i $MODEL_REPO_DIR/multimodal_encoders/config.pbtxt \
        triton_max_batch_size:8,visual_model_path:$VISION_ENGINE_DIR,encoder_input_features_data_type:TYPE_BF16,hf_model_path:$MODEL_DIR,max_queue_delay_microseconds:20000
fi

echo "running triton server..."

# exec python /app/tensorrtllm_backend/scripts/launch_triton_server.py --world_size 2 --model_repo $MODEL_REPO_DIR
# tritonserver --model-repository=$MODEL_REPO_DIR --log-verbose=1 --model-control-mode=explicit
tritonserver --model-repository=$MODEL_REPO_DIR --log-verbose=1 


# # Start Triton server with 2 GPUs
# python /app/tensorrtllm_backend/scripts/launch_triton_server.py \
#     --world_size 2 \
    # --model_repo $MODEL_REPO_DIR