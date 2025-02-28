#!/bin/bash

# Define directories for persistent storage with concrete paths
MODEL_DIR="/model_data/hf_models/Llama-3.2-11B-Vision"
CKPT_DIR="/model_data/trt_ckpts/Llama-3.2-11B-Vision"
LLM_ENGINE_DIR="/model_data/trt_engines/Llama-3.2-11B-Vision/llm/bf16/2-gpu"
VISION_ENGINE_DIR="/model_data/trt_engines/Llama-3.2-11B-Vision/vision/bf16/1-gpu"
MODEL_REPO_DIR="/app/model_data/multimodal_ifb"
# Create /app/model_data if it doesn't exist
mkdir -p "/app/model_data"

# Add checks to verify repository structure
echo "Checking TensorRT-LLM repository structure..."
ls -la "/app/tensorrt_llm"
echo "Checking tensorrtllm_backend all_models directory..."
ls -la "/app/tensorrtllm_backend/all_models"



# pip install tritonclient[http] tabulate
# pip install tabulate

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
        huggingface-cli download meta-llama/Llama-3.2-11B-Vision --local-dir "$MODEL_DIR"
        if [ $? -eq 0 ]; then
            echo "Model downloaded successfully."
            return 0
        else
            echo "Download failed. Retrying ($((retry_count + 1))/$max_retries)..."
            rm -rf "$MODEL_DIR"
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
# if [ ! -d "$LLM_ENGINE_DIR" ] || [ ! -d "$VISION_ENGINE_DIR" ]; then
#     echo "Building engines..."
#     python "/app/tensorrt_llm/examples/mllama/convert_checkpoint.py" \
#         --model_dir "$MODEL_DIR" \
#         --output_dir "$CKPT_DIR" \
#         --dtype bfloat16
#     trtllm-build \
#         --checkpoint_dir "$CKPT_DIR" \
#         --output_dir "$LLM_ENGINE_DIR" \
#         --max_num_tokens 4096 \
#         --max_seq_len 2048 \
#         --workers 1 \
#         --gemm_plugin auto \
#         --max_batch_size 1 \
#         --max_encoder_input_len 6404 \
#     python "/app/tensorrt_llm/examples/multimodal/build_visual_engine.py" \
#         --model_type mllama \
#         --model_path "$MODEL_DIR" \
#         --output_dir "$VISION_ENGINE_DIR" \
#         --max_batch_size 1
# fi

if [ ! -d "$LLM_ENGINE_DIR" ] || [ ! -d "$VISION_ENGINE_DIR" ]; then
    echo "Building engines..."
    python "/opt/tensorrt_llm/examples/mllama/convert_checkpoint.py" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
    /opt/tensorrt_llm/bin/trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$LLM_ENGINE_DIR" \
        --max_num_tokens 4096 \
        --max_seq_len 2048 \
        --gemm_plugin auto \
        --max_batch_size 1 \
        --max_encoder_input_len 6404
        
    python "/opt/tensorrt_llm/examples/multimodal/build_visual_engine.py" \
        --model_type mllama \
        --model_path "$MODEL_DIR" \
        --output_dir "$VISION_ENGINE_DIR" \
        --max_batch_size 1
fi

# Set up model repository
echo "Setting up model repository at $MODEL_REPO_DIR..."
rm -rf "$MODEL_REPO_DIR"  # Clear existing directory
mkdir -p "$MODEL_REPO_DIR"  # Ensure root directory exists

# Copy contents from multimodal and inflight_batcher_llm directly into MODEL_REPO_DIR
cp -r "/app/tensorrtllm_backend/all_models/multimodal/"* "$MODEL_REPO_DIR/"
cp -r "/app/tensorrtllm_backend/all_models/inflight_batcher_llm/"* "$MODEL_REPO_DIR/"

# Ensure version directories exist and copy engine files
mkdir -p "$MODEL_REPO_DIR/tensorrt_llm/1"
mkdir -p "$MODEL_REPO_DIR/multimodal_encoders/1"
cp "$LLM_ENGINE_DIR/rank0.engine" "$MODEL_REPO_DIR/tensorrt_llm/1/"
cp "$VISION_ENGINE_DIR/visual_encoder.engine" "$MODEL_REPO_DIR/multimodal_encoders/1/"

# Fill templates with concrete paths
# python "/app/tensorrtllm_backend/tools/fill_template.py" -i "$MODEL_REPO_DIR/preprocessing/config.pbtxt" \
#     tokenizer_dir:"$MODEL_DIR",triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:"$VISION_ENGINE_DIR",engine_dir:"$LLM_ENGINE_DIR",max_num_images:1,max_queue_delay_microseconds:20000
# python "/app/tensorrtllm_backend/tools/fill_template.py" -i "$MODEL_REPO_DIR/tensorrt_llm/config.pbtxt" \
#     triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:"$LLM_ENGINE_DIR",enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:TYPE_BF16,logits_datatype:TYPE_FP32,cross_kv_cache_fraction:0.5
# python "/app/tensorrtllm_backend/tools/fill_template.py" -i "$MODEL_REPO_DIR/multimodal_encoders/config.pbtxt" \
#     triton_max_batch_size:8,visual_model_path:"$VISION_ENGINE_DIR",encoder_input_features_data_type:TYPE_BF16,hf_model_path:"$MODEL_DIR",max_queue_delay_microseconds:20000
# python "/app/tensorrtllm_backend/tools/fill_template.py" -i "$MODEL_REPO_DIR/postprocessing/config.pbtxt" \
#     tokenizer_dir:"$MODEL_DIR",triton_max_batch_size:8,postprocessing_instance_count:1
# python "/app/tensorrtllm_backend/tools/fill_template.py" -i "$MODEL_REPO_DIR/ensemble/config.pbtxt" \
#     triton_max_batch_size:8,logits_datatype:TYPE_FP32
# python "/app/tensorrtllm_backend/tools/fill_template.py" -i "$MODEL_REPO_DIR/tensorrt_llm_bls/config.pbtxt" \
#     triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32


ENCODER_INPUT_FEATURES_DTYPE=TYPE_BF16
python3 /app/tensorrtllm_backend/tools/fill_template.py -i "$MODEL_REPO_DIR/tensorrt_llm/config.pbtxt" triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${LLM_ENGINE_DIR},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},logits_datatype:TYPE_FP32,cross_kv_cache_fraction:0.5

python3 /app/tensorrtllm_backend/tools/fill_template.py -i "$MODEL_REPO_DIR/preprocessing/config.pbtxt" tokenizer_dir:${MODEL_DIR},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:${VISION_ENGINE_DIR},engine_dir:${LLM_ENGINE_DIR},max_num_images:1

python3 /app/tensorrtllm_backend/tools/fill_template.py -i "$MODEL_REPO_DIR/postprocessing/config.pbtxt" tokenizer_dir:${MODEL_DIR},triton_max_batch_size:8,postprocessing_instance_count:1

python3 /app/tensorrtllm_backend/tools/fill_template.py -i "$MODEL_REPO_DIR/ensemble/config.pbtxt" triton_max_batch_size:8,logits_datatype:TYPE_FP32

python3 /app/tensorrtllm_backend/tools/fill_template.py -i "$MODEL_REPO_DIR/tensorrt_llm_bls/config.pbtxt" triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32

# Newly added for multimodal
python3 /app/tensorrtllm_backend/tools/fill_template.py -i "$MODEL_REPO_DIR/multimodal_encoders/config.pbtxt" triton_max_batch_size:8,visual_model_path:${VISION_ENGINE_DIR},encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},hf_model_path:${MODEL_DIR}




# Verify configs after filling templates
# echo "Verifying preprocessing config..."
# cat "$MODEL_REPO_DIR/preprocessing/config.pbtxt"
# echo "Verifying ensemble config..."
# cat "$MODEL_REPO_DIR/ensemble/config.pbtxt"

# Verify the structure
echo "Model repository structure:"
ls -la "$MODEL_REPO_DIR"
echo $MODEL_REPO_DIR
ls -la /app

apt-get install -y tree
tree $MODEL_REPO_DIR

export PMIX_MCA_gds=hash

export CUDA_VISIBLE_DEVICES=0,1  # Ensure both GPUs are available


# python3 /app/scripts/launch_triton_server.py --world_size 1 --model_repo=$MODEL_REPO_DIR/ --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders --multimodal_gpu0_cuda_mem_pool_bytes 300000000



tritonserver \
    --model-repository="$MODEL_REPO_DIR" \
    --grpc-port=8001 \
    --http-port=8000 \
    --metrics-port=8002 \
    # --disable-auto-complete-config \
    # --cuda-memory-pool-byte-size=0:100000000 \
    # --cuda-memory-pool-byte-size=1:100000000 \
    # --multimodal_gpu0_cuda_mem_pool_bytes 100000000 \
    --multimodal_gpu0_cuda_mem_pool_bytes 300000000
    # --model-control-mode=none \
    --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders \
    -- world_size 2 
TRITON_PID=$!

# Wait for server to start
sleep 10

# Verify structure after server start
ls -la "$MODEL_REPO_DIR"

# Keep container running
wait $TRITON_PID
echo "Triton server started and running."


# # Start Triton server in background and load the ensemble model
# echo "Starting Triton server..."
# tritonserver --model-repository="$MODEL_REPO_DIR" --log-verbose=1 --model-control-mode=explicit &
# TRITON_PID=$!

# # Wait for server to start (adjust sleep time as needed)
# sleep 10

# # Explicitly load the ensemble model
# echo "Loading ensemble model..."
# curl -X POST http://localhost:8000/v2/repository/models/ensemble/load || { echo "Failed to load ensemble model"; kill $TRITON_PID; exit 1; }

# # Keep container running
# wait $TRITON_PID
# echo "Triton server started and ensemble model loaded."


# Start Triton server
# echo "Starting Triton server..."
# # tritonserver --model-repository="$MODEL_REPO_DIR" --log-verbose=1 --model-control-mode=explicit & TRITON_PID=$!
# tritonserver --model-repository="$MODEL_REPO_DIR" --log-verbose=1 --model-control-mode=none & TRITON_PID=$!
# # python "/app/tensorrtllm_backend/scripts/launch_triton_server.py" \
# #     --world_size 2 \
# #     --model_repo "$MODEL_REPO_DIR" || { echo "Triton server failed to start with exit code $?"; exit 1; }
# # exec python /app/tensorrtllm_backend/scripts/launch_triton_server.py --world_size 2 --model_repo $MODEL_REPO_DIR
# # tritonserver --model-repository=$MODEL_REPO_DIR --log-verbose=1 --model-control-mode=explicit
# # tritonserver --model-repository=$MODEL_REPO_DIR --log-verbose=1 
# wait $TRITON_PID


# # # Start Triton server with 2 GPUs
# # python /app/tensorrtllm_backend/scripts/launch_triton_server.py \
# #     --world_size 2 \
#     # --model_repo $MODEL_REPO_DIR
# echo "Triton server started."

