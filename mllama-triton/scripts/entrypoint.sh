#!/bin/bash
set -e

# Check if model_engine exists and has engines
if [ ! -d "/model_engine/vision" ] || [ ! -d "/model_engine/llm" ]; then
    echo "Model engines not found, building them now..."
    cd /app
    chmod +x build_engines.py
    python3 build_engines.py --model_path /models/Llama-3.2-11B-Vision --output_dir /model_engine
    echo "Engines built successfully!"
else
    echo "Model engines already exist, skipping build."
fi

# Start Triton Server
echo "Starting Triton Server..."
exec tritonserver --model-repository=/models/multimodal_ifb --log-verbose=1