#!/bin/bash
set -e

# Force UCX library path priority
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# Default model if not specified
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-11B-Vision-Instruct"}
MODEL_DIR="/models/$(echo "$MODEL_NAME" | sed 's/\//-/g')"

# Map model to quantization for engine building
case "$MODEL_NAME" in
    "meta-llama/Llama-3.2-11B-Vision-Instruct")
        QUANTization="int4"  # Default to INT4 for original model
        ;;
    "neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic")
        QUANTization="fp8"
        ;;
    "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit")
        QUANTization="int4"
        ;;
    *)
        echo "Unsupported model: $MODEL_NAME. Using default: meta-llama/Llama-3.2-11B-Vision-Instruct"
        MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
        QUANTization="int4"
        ;;
esac

if [ ! -d "/models/multimodal_ifb" ]; then
    echo "Creating and populating /models/multimodal_ifb..."
    mkdir -p /models/multimodal_ifb
    cd /app/tensorrtllm_backend/tools
    python3 fill_template.py \
        -i /models/multimodal_ifb/tensorrt_llm/config.pbtxt \
        triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:/models/tensorrt_llm/1/,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:TYPE_${QUANTization^^},logits_datatype:TYPE_FP32
    python3 fill_template.py \
        -i /models/multimodal_ifb/preprocessing/config.pbtxt \
        tokenizer_dir:${MODEL_DIR},triton_max_batch_size:8,preprocessing_instance_count:1,visual_model_path:/models/multimodal_encoders/1/,engine_dir:/models/tensorrt_llm/1/,max_num_images:1
    python3 fill_template.py \
        -i /models/multimodal_ifb/postprocessing/config.pbtxt \
        tokenizer_dir:${MODEL_DIR},triton_max_batch_size:8,postprocessing_instance_count:1
    python3 fill_template.py \
        -i /models/multimodal_ifb/ensemble/config.pbtxt \
        triton_max_batch_size:8,logits_datatype:TYPE_FP32
    python3 fill_template.py \
        -i /models/multimodal_ifb/tensorrt_llm_bls/config.pbtxt \
        triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32
    python3 fill_template.py \
        -i /models/multimodal_ifb/multimodal_encoders/config.pbtxt \
        triton_max_batch_size:8,visual_model_path:/models/multimodal_encoders/1/,encoder_input_features_data_type:TYPE_${QUANTization^^},hf_model_path:${MODEL_DIR}
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
        echo "Building engines for $MODEL_NAME with $QUANTization quantization..."
        chmod +x build_engines.py
        python3 build_engines.py --model_path "$MODEL_DIR" --output_dir /model_engine --max_batch_size 1 --tp_size 2 --quantization "$QUANTization"
        echo "Engines built successfully!"
    else
        echo "Engines already exist, skipping build."
    fi
else
    echo "Model $MODEL_NAME and engines already exist, skipping setup."
fi

echo "Starting Triton Server..."
exec tritonserver --model-repository=/models/multimodal_ifb --log-verbose=1