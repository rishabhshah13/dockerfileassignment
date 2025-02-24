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