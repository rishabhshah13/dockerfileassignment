#!/bin/bash
set -e

# Force UCX library path priority
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Check if model and engines exist
if [ ! -d "/models/Llama-3.2-11B-Vision" ] || [ ! -d "/model_engine/vision" ] || [ ! -d "/model_engine/llm" ]; then
    echo "Model or engines not found, setting up now..."
    cd /app
    if [ ! -d "/models/Llama-3.2-11B-Vision" ]; then
        echo "Downloading LLaMA 3.2 11B Vision model..."
        mkdir -p /models/Llama-3.2-11B-Vision
        huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct --local-dir /models/Llama-3.2-11B-Vision --token "$HF_TOKEN"
    else
        echo "Model already exists, skipping download."
    fi
    if [ ! -d "/model_engine/vision" ] || [ ! -d "/model_engine/llm" ]; then
        echo "Building engines..."
        chmod +x build_engines.py
        python3 build_engines.py --model_path /models/Llama-3.2-11B-Vision --output_dir /model_engine
        echo "Engines built successfully!"
    else
        echo "Engines already exist, skipping build."
    fi
else
    echo "Model and engines already exist, skipping setup."
fi

echo "Starting Triton Server..."
exec tritonserver --model-repository=/models/multimodal_ifb --log-verbose=1