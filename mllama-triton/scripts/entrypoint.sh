#!/bin/bash
set -e

# Default model if not specified
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-11B-Vision-Instruct"}
MODEL_DIR="/models/$(echo "$MODEL_NAME" | sed 's/\//-/g')"

# Map model to quantization and data types for engine building
case "$MODEL_NAME" in
    "meta-llama/Llama-3.2-11B-Vision-Instruct")
        QUANTIZATION="int4"
        ENCODER_INPUT_FEATURES_DTYPE="TYPE_INT4"
        ;;
    "neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic")
        QUANTIZATION="fp8"
        ENCODER_INPUT_FEATURES_DTYPE="TYPE_FP8"
        ;;
    "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit")
        QUANTIZATION="int4"
        ENCODER_INPUT_FEATURES_DTYPE="TYPE_INT4"
        ;;
    *)
        echo "Unsupported model: $MODEL_NAME. Using default: meta-llama/Llama-3.2-11B-Vision-Instruct"
        MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
        QUANTIZATION="int4"
        ENCODER_INPUT_FEATURES_DTYPE="TYPE_INT4"
        ;;
esac

# Set paths for engines
ENGINE_PATH="/models/tensorrt_llm/1/"
VISUAL_ENGINE_PATH="/models/multimodal_encoders/1/"
HF_MODEL_PATH="${MODEL_DIR}"

# Ensure /models/multimodal_ifb exists and is populated dynamically
if [ -z "$(ls -A /models/multimodal_ifb)" ]; then
    echo "Populating /models/multimodal_ifb for $MODEL_NAME with $QUANTIZATION quantization..."
    mkdir -p /models/multimodal_ifb
    
    # Copy base structures from tensorrtllm_backend/all_models, ensuring correct subdirectories
    cd /app/tensorrtllm_backend
    cp -r all_models/inflight_batcher_llm/tensorrt_llm /models/multimodal_ifb/
    cp -r all_models/multimodal/ensemble /models/multimodal_ifb/
    cp -r all_models/multimodal/multimodal_encoders /models/multimodal_ifb/

    # Create empty .pbtxt files if they don’t exist to ensure fill_template.py can modify them
    # mkdir -p /models/multimodal_ifb/tensorrt_llm
    # mkdir -p /models/multimodal_ifb/preprocessing
    # mkdir -p /models/multimodal_ifb/postprocessing
    # mkdir -p /models/multimodal_ifb/ensemble
    # mkdir -p /models/multimodal_ifb/tensorrt_llm_bls
    # mkdir -p /models/multimodal_ifb/multimodal_encoders
    # touch /models/multimodal_ifb/tensorrt_llm/config.pbtxt
    # touch /models/multimodal_ifb/preprocessing/config.pbtxt
    # touch /models/multimodal_ifb/postprocessing/config.pbtxt
    # touch /models/multimodal_ifb/ensemble/config.pbtxt
    # touch /models/multimodal_ifb/tensorrt_llm_bls/config.pbtxt
    # touch /models/multimodal_ifb/multimodal_encoders/config.pbtxt

    # Verify the directories and files are readable
    if [ ! -r /models/multimodal_ifb ] || [ ! -r /models/multimodal_ifb/* ]; then
        echo "Error: /models/multimodal_ifb or its contents are not readable. Fixing permissions..."
        chmod -R +r /models/multimodal_ifb
    fi

    # Verify fill_template.py exists and is executable in /app/tensorrtllm_backend/tools
    if [ ! -f "/app/tensorrtllm_backend/tools/fill_template.py" ]; then
        echo "Error: fill_template.py not found at /app/tensorrtllm_backend/tools. Checking clone..."
        ls -l /app/tensorrtllm_backend/tools || echo "Tools directory missing or empty."
        exit 1
    fi
    if [ ! -x "/app/tensorrtllm_backend/tools/fill_template.py" ]; then
        echo "Making fill_template.py executable..."
        chmod +x /app/tensorrtllm_backend/tools/fill_template.py
    fi

    # Run fill_template.py to generate Triton configs with error checking
    cd /app/tensorrtllm_backend/tools
    python3 fill_template.py \
        -i /models/multimodal_ifb/tensorrt_llm/config.pbtxt \
        triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},logits_datatype:TYPE_FP32 || echo "Failed to generate tensorrt_llm config"
    python3 fill_template.py \
        -i /models/multimodal_ifb/preprocessing/config.pbtxt \
        tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:${VISUAL_ENGINE_PATH},engine_dir:${ENGINE_PATH},max_num_images:1 || echo "Failed to generate preprocessing config"
    python3 fill_template.py \
        -i /models/multimodal_ifb/postprocessing/config.pbtxt \
        tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:8,postprocessing_instance_count:1 || echo "Failed to generate postprocessing config"
    python3 fill_template.py \
        -i /models/multimodal_ifb/ensemble/config.pbtxt \
        triton_max_batch_size:8,logits_datatype:TYPE_FP32 || echo "Failed to generate ensemble config"
    python3 fill_template.py \
        -i /models/multimodal_ifb/tensorrt_llm_bls/config.pbtxt \
        triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32 || echo "Failed to generate tensorrt_llm_bls config"
    python3 fill_template.py \
        -i /models/multimodal_ifb/multimodal_encoders/config.pbtxt \
        triton_max_batch_size:8,visual_model_path:${VISUAL_ENGINE_PATH},encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},hf_model_path:${HF_MODEL_PATH} || echo "Failed to generate multimodal_encoders config"

    # Verify the directory is populated
    if [ -z "$(ls -A /models/multimodal_ifb)" ]; then
        echo "Error: /models/multimodal_ifb is still empty after population attempt. Check cp commands, fill_template.py, or mounts."
        exit 1
    fi
else
    echo "/models/multimodal_ifb is already populated."
fi

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
        echo "Building engines for $MODEL_NAME with $QUANTIZATION quantization..."
        chmod +x build_engines.py
        python3 build_engines.py --model_path "$MODEL_DIR" --output_dir /model_engine --max_batch_size 1 --tp_size 2 --quantization "$QUANTIZATION"
        echo "Engines built successfully!"
    else
        echo "Engines already exist, skipping build."
    fi
else
    echo "Model $MODEL_NAME and engines already exist, skipping setup."
fi

echo "Starting Triton Server..."
exec tritonserver --model-repository=/models/multimodal_ifb --log-verbose=1