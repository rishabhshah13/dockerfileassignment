#!/bin/bash

# Define paths as variables
# TENSORRT_LLM_BACKEND_DIR="/app/tensorrtllm_backend"
# TENSORRT_LLM_DIR="/app/tensorrtllm_backend/tensorrt_llm"
# TENSORRT_LLM_BACKEND_DIR="/app"
# TENSORRT_LLM_DIR="/app"
HF_MODEL_PATH="/model_data/hf_models/Llama-3.2-11B-Vision"
CKPT_DIR="/model_data/trt_ckpts/Llama-3.2-11B-Vision"
# ENGINE_PATH="/model_data/trt_engines/Llama-3.2-11B-Vision/llm/bf16/2-gpu"
ENGINE_PATH="/model_data/trt_engines/Llama-3.2-11B-Vision/llm/int8/2-gpu"
VISUAL_ENGINE_PATH="/model_data/trt_engines/Llama-3.2-11B-Vision/vision/bf16/1-gpu"
MODEL_REPO_DIR="/app/model_data/multimodal_ifb"

# Create /app/model_data if it doesn't exist
mkdir -p "/app/model_data"

# Check repository structure
# echo "Checking TensorRT-LLM repository structure..."
# ls -la "$TENSORRT_LLM_DIR"
# echo "Checking tensorrtllm_backend all_models directory..."
# ls -la "$TENSORRT_LLM_BACKEND_DIR/all_models"

# Login to Hugging Face if token is provided
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN environment variable not set. Please provide your Hugging Face token."
    exit 1
fi

# Function to download model with retry
download_model() {
    local max_retries=3
    local retry_count=0
    while [ $retry_count -lt $max_retries ]; do
        huggingface-cli download meta-llama/Llama-3.2-11B-Vision --local-dir "$HF_MODEL_PATH"
        if [ $? -eq 0 ]; then
            echo "Model downloaded successfully."
            return 0
        else
            echo "Download failed. Retrying ($((retry_count + 1))/$max_retries)..."
            rm -rf "$HF_MODEL_PATH"
            retry_count=$((retry_count + 1))
            sleep 10
        fi
    done
    echo "Failed to download model after $max_retries attempts."
    exit 1
}

# Download model if not already present
if [ ! -d "$HF_MODEL_PATH" ] || [ ! -f "$HF_MODEL_PATH/original/consolidated.pth" ]; then
    echo "Downloading model..."
    download_model
fi

echo $ENGINE_PATH
echo $VISUAL_ENGINE_PATH

ls -la "$ENGINE_PATH"
ls -la "$VISUAL_ENGINE_PATH"

# Build engines if not already built
if [ ! -d "$ENGINE_PATH" ] || [ ! -d "$VISUAL_ENGINE_PATH" ]; then
    echo "Building engines..."

    # python "/app/examples/mllama/convert_checkpoint.py" \
    #     --model_dir "$HF_MODEL_PATH" \
    #     --output_dir "$CKPT_DIR" \
    #     --dtype bfloat16 \
    #     --workers 2
    
    # python "/app/examples/quantization/quantize.py" \
    #     --model_dir "$HF_MODEL_PATH" \
    #     --output_dir "$CKPT_DIR_INT8" \
    #     --dtype bfloat16 \
    #     --qformat int8_sq \
    #     # --calib_size 32 \
    #     # --calib_dataset scienceqa

    # "trtllm-build" \
    #     --checkpoint_dir "$CKPT_DIR_INT8" \
    #     --output_dir "$ENGINE_PATH" \
    #     --max_num_tokens 4096 \
    #     --max_seq_len 2048 \
    #     --gemm_plugin auto \
    #     --max_batch_size 1 \
    #     --max_encoder_input_len 6404 \
    #     --workers 2
            
    # python "/app/examples/multimodal/build_visual_engine.py" \
    #     --model_type mllama \
    #     --model_path "$HF_MODEL_PATH" \
    #     --output_dir "$VISUAL_ENGINE_PATH" \
    #     --max_batch_size 1

    # 1. Convert HF checkpoint to TRT-LLM checkpoint with INT8 weight-only quantization.
    python "/app/examples/mllama/convert_checkpoint.py" \
        --model_dir "$HF_MODEL_PATH" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16 \
        --use_weight_only \
        --weight_only_precision int8 \
        --workers 2

    # 2. Build the TensorRT engines using the INT8 weight-only checkpoint.
    trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$ENGINE_PATH" \
        --max_num_tokens 2048 \
        --max_seq_len 2048 \
        --gemm_plugin auto \
        --max_batch_size 1 \
        --max_encoder_input_len 6404 \
        --workers 2 

    # 3. Build the visual engine (for the vision encoder part) for MLLaMA.
    python "/app/examples/multimodal/build_visual_engine.py" \
        --model_type mllama \
        --model_path "$HF_MODEL_PATH" \
        --output_dir "$VISUAL_ENGINE_PATH" \
        --max_batch_size 1


fi

# Set up model repository
echo "Setting up model repository at $MODEL_REPO_DIR..."
rm -rf "$MODEL_REPO_DIR"
mkdir -p "$MODEL_REPO_DIR"

# Copy models
cp -r "/app/all_models/inflight_batcher_llm/"* "$MODEL_REPO_DIR/" 
cp "/app/all_models/multimodal/ensemble" "$MODEL_REPO_DIR/"  -r
cp "/app/all_models/multimodal/multimodal_encoders" "$MODEL_REPO_DIR/"  -r

# Ensure directories exist
mkdir -p "$MODEL_REPO_DIR/tensorrt_llm/1"
mkdir -p "$MODEL_REPO_DIR/multimodal_encoders/1"
cp "$ENGINE_PATH/*" "$MODEL_REPO_DIR/tensorrt_llm/1/"
cp "$VISUAL_ENGINE_PATH/*" "$MODEL_REPO_DIR/multimodal_encoders/1/"



# Fill templates
ENCODER_INPUT_FEATURES_DTYPE=TYPE_BF16

# python3 "$TENSORRT_LLM_BACKEND_DIR/tools/fill_template.py" -i "$MODEL_REPO_DIR/tensorrt_llm/config.pbtxt" triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${LLM_ENGINE_DIR},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},logits_datatype:TYPE_FP32,cross_kv_cache_fraction:0.5

# python3 "$TENSORRT_LLM_BACKEND_DIR/tools/fill_template.py" -i "$MODEL_REPO_DIR/preprocessing/config.pbtxt" tokenizer_dir:${MODEL_DIR},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:${VISION_ENGINE_DIR},engine_dir:${LLM_ENGINE_DIR},max_num_images:1

# python3 "$TENSORRT_LLM_BACKEND_DIR/tools/fill_template.py" -i "$MODEL_REPO_DIR/postprocessing/config.pbtxt" tokenizer_dir:${MODEL_DIR},triton_max_batch_size:8,postprocessing_instance_count:1

# python3 "$TENSORRT_LLM_BACKEND_DIR/tools/fill_template.py" -i "$MODEL_REPO_DIR/ensemble/config.pbtxt" triton_max_batch_size:8,logits_datatype:TYPE_FP32

# python3 "$TENSORRT_LLM_BACKEND_DIR/tools/fill_template.py" -i "$MODEL_REPO_DIR/tensorrt_llm_bls/config.pbtxt" triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32

# # Newly added for multimodal
# python3 "$TENSORRT_LLM_BACKEND_DIR/tools/fill_template.py" -i "$MODEL_REPO_DIR/multimodal_encoders/config.pbtxt" triton_max_batch_size:8,visual_model_path:${VISION_ENGINE_DIR},encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},MODEL_DIR:${HF_MODEL_PATH}


python3 /app/tools/fill_template.py -i ${MODEL_REPO_DIR}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},logits_datatype:TYPE_FP32,cross_kv_cache_fraction:0.02

python3 /app/tools/fill_template.py -i ${MODEL_REPO_DIR}/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:${VISUAL_ENGINE_PATH},engine_dir:${ENGINE_PATH},max_num_images:1

python3 /app/tools/fill_template.py -i ${MODEL_REPO_DIR}/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,postprocessing_instance_count:1

python3 /app/tools/fill_template.py -i ${MODEL_REPO_DIR}/ensemble/config.pbtxt triton_max_batch_size:8,logits_datatype:TYPE_FP32

python3 /app/tools/fill_template.py -i ${MODEL_REPO_DIR}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32

# Newly added for multimodal
python3 /app/tools/fill_template.py -i ${MODEL_REPO_DIR}/multimodal_encoders/config.pbtxt triton_max_batch_size:8,visual_model_path:${VISUAL_ENGINE_PATH},encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},hf_model_path:${HF_MODEL_PATH}


# Verify model repo structure
echo "Model repository structure:"
ls -la "$MODEL_REPO_DIR"
ls -la /app

# Install tree for better directory visualization
apt-get install -y tree
tree "$MODEL_REPO_DIR"

export PMIX_MCA_gds=hash
export CUDA_VISIBLE_DEVICES=0,1

# Start Triton Server
# tritonserver \
#     --model-repository="$MODEL_REPO_DIR" \
#     --grpc-port=8001 \
#     --http-port=8000 \
#     --metrics-port=8002 
#     # --log-verbose 1 
#     # --world_size 2 \
#     # --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders \


export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

echo "Starting Triton server with 2 MPI processes for LLM split across GPUs..."
mpirun -np 2 --map-by socket --allow-run-as-root \
    -x PMIX_MCA_gds=hash \
    -x CUDA_VISIBLE_DEVICES=0,1 \
    tritonserver \
        --model-repository="$MODEL_REPO_DIR" \
        --grpc-port=8001 \
        --http-port=8000 \
        --metrics-port=8002 \
        --log-verbose 1 &

TRITON_PID=$!

# Wait for server to start
sleep 10

# Verify structure after server start
ls -la "$MODEL_REPO_DIR"

# Keep container running
wait $TRITON_PID
echo "Triton server started and running."
