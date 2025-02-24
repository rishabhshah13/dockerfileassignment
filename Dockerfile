# Stage 1: Build Stage - Model Engine Creation
FROM nvcr.io/nvidia/tensorrt:23.10-py3 AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y git wget
RUN pip install --upgrade pip
RUN pip install huggingface_hub

# Clone TensorRT-LLM
RUN git clone https://github.com/NVIDIA/TensorRT-LLM /TensorRT-LLM
WORKDIR /TensorRT-LLM
RUN pip install -r requirements.txt

# Download Llama 3.2 11B Vision model
RUN mkdir -p /models/Llama-3.2-11B-Vision
RUN huggingface-cli download meta-llama/Llama-3.2-11B-Vision --local-dir /models/Llama-3.2-11B-Vision

# Build the visual engine with INT8 precision
WORKDIR /TensorRT-LLM/examples/multimodal
RUN python build_visual_engine.py --model_type mllama \
    --model_path /models/Llama-3.2-11B-Vision \
    --output_dir /tmp/mllama/trt_engines/encoder/ \
    --dtype int8

# Stage 2: Triton Backend Build Stage
FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS triton-builder
WORKDIR /app

RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend /tensorrtllm_backend
WORKDIR /tensorrtllm_backend
RUN ./build.sh

# Stage 3: Runtime Stage
FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS runtime
WORKDIR /app

# Copy model engine and backend
COPY --from=builder /tmp/mllama/trt_engines/encoder/ /model_engine
COPY --from=triton-builder /tensorrtllm_backend/build/tensorrtllm_backend.so /opt/tritonserver/backends/tensorrtllm_backend/libtensorrtllm_backend.so

# Configure model
RUN mkdir -p /opt/tritonserver/models/mllama/1
COPY model_config.pbtxt /opt/tritonserver/models/mllama/1/config.pbtxt
RUN cp -r /model_engine /opt/tritonserver/models/mllama/1

# Start Triton server
CMD ["tritonserver", "--model-store=/opt/tritonserver/models", "--backend-config=tensorrtllm_backend,config.pb"]