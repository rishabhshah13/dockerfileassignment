#!/bin/bash
set -e

# Default model for MLLaMA
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-11B-Vision"}
MODEL_DIR="/models/$(echo "$MODEL_NAME" | sed 's/\//-/g')"

# For MLLaMA, use bfloat16
ENCODER_INPUT_FEATURES_DTYPE="TYPE_INT4"
# ENCODER_INPUT_FEATURES_DTYPE="TYPE_BF16"

# Set paths for models and engines
ENGINE_PATH="/models/tensorrt_llm/1/"
VISUAL_ENGINE_PATH="/models/multimodal_encoders/1/"
HF_MODEL_PATH="${MODEL_DIR}"


# Ensure /models/multimodal_ifb exists and is populated
echo "Setting up Triton model repository..."
mkdir -p /models/multimodal_ifb

# Copy base structures from tensorrtllm_backend/all_models
cd /app/tensorrtllm_backend
cp -r all_models/inflight_batcher_llm/* /models/multimodal_ifb/
cp -r all_models/multimodal/ensemble /models/multimodal_ifb/
cp -r all_models/multimodal/multimodal_encoders /models/multimodal_ifb/

# Check if model and engines exist
if [ ! -d "$MODEL_DIR" ] || [ ! -d "/model_engine/vision" ] || [ ! -d "/model_engine/llm" ]; then
    echo "Model or engines not found, setting up now..."
    cd /app
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Downloading $MODEL_NAME..."
        mkdir -p "$MODEL_DIR"
        huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_DIR" --token "$HF_TOKEN"
    else
        echo "Model $MODEL_NAME already exists, skipping download."
    fi
    
    if [ ! -d "/model_engine/vision" ] || [ ! -d "/model_engine/llm" ]; then
        echo "Building engines for MLLaMA..."
        python3 build_engines.py --model_path "$MODEL_DIR" --output_dir /model_engine --max_batch_size 1
        echo "Engines built successfully!"
    else
        echo "Engines already exist, skipping build."
    fi
fi

# Prepare engine directories for Triton
mkdir -p ${ENGINE_PATH}
mkdir -p ${VISUAL_ENGINE_PATH}

# Copy built engines to Triton model directories if they exist
if [ -d "/model_engine/llm" ]; then
    echo "Copying LLM engine to ${ENGINE_PATH}..."
    cp -r /model_engine/llm/* ${ENGINE_PATH}/
fi

if [ -d "/model_engine/vision" ]; then
    echo "Copying vision engine to ${VISUAL_ENGINE_PATH}..."
    cp -r /model_engine/vision/* ${VISUAL_ENGINE_PATH}/
fi

# Generate Triton model configuration
cd /app/tensorrtllm_backend/tools
echo "Generating Triton model configurations..."

python3 fill_template.py -i /models/multimodal_ifb/tensorrt_llm/config.pbtxt \
    triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},logits_datatype:TYPE_FP32,cross_kv_cache_fraction:0.5

python3 fill_template.py -i /models/multimodal_ifb/preprocessing/config.pbtxt \
    tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:${VISUAL_ENGINE_PATH},engine_dir:${ENGINE_PATH},max_num_images:1,max_queue_delay_microseconds:20000

python3 fill_template.py -i /models/multimodal_ifb/postprocessing/config.pbtxt \
    tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,postprocessing_instance_count:1

python3 fill_template.py -i /models/multimodal_ifb/ensemble/config.pbtxt \
    triton_max_batch_size:8,logits_datatype:TYPE_FP32

python3 fill_template.py -i /models/multimodal_ifb/tensorrt_llm_bls/config.pbtxt \
    triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32

python3 fill_template.py -i /models/multimodal_ifb/multimodal_encoders/config.pbtxt \
    triton_max_batch_size:8,visual_model_path:${VISUAL_ENGINE_PATH},encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},hf_model_path:${HF_MODEL_PATH},max_queue_delay_microseconds:20000

# Start Triton Server
echo "Starting Triton Server..."
# export PMIX_MCA_gds=hash  # To avoid MPI_Init_thread errors
tritonserver --model-repository=/models/multimodal_ifb --log-verbose=1 --model-control-mode=explicit
