# Stage 1: Build Stage: Model Engine Creation
FROM nvcr.io/nvidia/tensorrt:8.6.1-py3 as builder
WORKDIR /app

RUN apt-get update && apt-get install -y git wget
RUN pip install huggingface_hub

RUN git clone https://github.com/NVIDIA/TensorRT-LLM /TensorRT-LLM
RUN pip install -r /TensorRT-LLM/requirements.txt

RUN mkdir -p /models/Llama-3.2-11B-Vision
RUN huggingface-cli download meta-llama/Llama-3.2-11B-Vision --local-dir /models/Llama-3.2-11B-Vision

WORKDIR /TensorRT-LLM/examples/multimodal
RUN python build_visual_engine.py --model_type mllama \
    --model_path /models/Llama-3.2-11B-Vision \
    --output_dir /tmp/mllama/trt_engines/encoder/ \
    --dtype int8

# Stage 2: Triton Backend Build Stage
FROM nvcr.io/nvidia/tritonserver:24.01-py3 as triton-builder
WORKDIR /app

RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend /tensorrtllm_backend
WORKDIR /tensorrtllm_backend
RUN ./build.sh


# Stage 3: Runtime Stage
FROM nvcr.io/nvidia/tritonserver:24.01-py3 as runtime
WORKDIR /app

COPY --from=builder /tmp/mllama/trt_engines/encoder/ /model_engine
COPY --from=triton-builder /tensorrtllm_backend/build/tensorrtllm_backend.so /opt/tritonserver/backends/tensorrtllm_backend/libtensorrtllm_backend.so

RUN mkdir -p /opt/tritonserver/models/mllama/1
COPY model_config.pbtxt /opt/tritonserver/models/mllama/1/config.pbtxt
RUN cp -r /model_engine /opt/tritonserver/models/mllama/1

CMD ["tritonserver", "--model-store=/opt/tritonserver/models", "--backend-config=tensorrtllm_backend,config.pb"]