

docker build -t llama-triton:latest --build-arg HF_TOKEN=$HF_TOKEN .


docker build -t llama-triton:latest --build-arg HF_TOKEN=YOUR_HF_TOKEN . --progress=plain

docker build -t llama-triton:latest --build-arg HF_TOKEN=YOUR_HF_TOKEN . 

mkdir -p /home/rishabhshah/llama_engine
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/rishabhshah/llama_engine:/model_engine llama-triton:latest


huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct --local-dir ~/models/Llama-3.2-11B-Vision --token YOUR_HF_TOKEN

 

docker login nvcr.io


Username: $oauthtoken
Password: 


mkdir -p llama_engine
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v llama_engine:/model_engine llama-triton:latest



docker build -t mllama-triton:latest --build-arg HF_TOKEN=YOUR_HF_TOKEN .

docker buildx build --build-arg HF_TOKEN=YOUR_HF_TOKEN -t mllama-triton:latest --gpus all .



docker build -t mllama-triton:latest --build-arg HF_TOKEN=YOUR_HF_TOKEN .




docker run --gpus all -v $(pwd)/model_engine:/app/model_engine -v $(pwd)/models:/app/models -p 8000:8000 -p 8001:8001 -p 8002:8002 -e MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct" mllama-triton:latest

docker run --gpus all -v $(pwd)/model_engine:/app/model_engine -v $(pwd)/models:/app/models -p 8000:8000 -p 8001:8001 -p 8002:8002 -e MODEL_NAME="neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic" mllama-triton:latest

docker run --gpus all -v $(pwd)/model_engine:/app/model_engine -v $(pwd)/models:/app/models -p 8000:8000 -p 8001:8001 -p 8002:8002 -e MODEL_NAME="unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit" mllama-triton:latest


docker run --rm -it --gpus all --entrypoint /bin/bash \
    -v $(pwd)/model_engine:/app/model_engine \
    -v $(pwd)/models:/app/models \
    mllama-triton:latest \
    -p 8000:8000 -p 8001:8001 -p 8002:8002

# exec into running one
docker exec -it ee3bd5913068 /bin/bash

tritonserver --model-repository=/models/multimodal_ifb --log-verbose=1 --model-control-mode=explicit


docker run --rm -it --gpus all --entrypoint /bin/bash \
    mllama-triton:latest \
    -p 8000:8000 -p 8001:8001 -p 8002:8002


docker run --rm -it --gpus all --entrypoint /bin/bash \
    llama_vision_triton \
    -p 8000:8000 -p 8001:8001 -p 8002:8002

# docker build --build-arg HF_TOKEN=YOUR_HF_TOKEN -t triton_mllama .
# docker run --rm -it --gpus all -p 8000:8000 triton_mllama

# ARG HF_TOKEN
RUN mkdir -p /models/Llama-3.2-11B-Vision && \
    huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
    --local-dir /models/Llama-3.2-11B-Vision --token YOUR_HF_TOKEN



#running some

docker build -t llama_vision_triton .

docker run --gpus all -v /mnt/model_data:/model_data -e HF_TOKEN=YOUR_HF_TOKEN -p 8000:8000 -p 8001:8001 -p 8002:8002 llama_vision_triton



curl -X POST localhost:8000/v2/repository/index
curl -X POST localhost:8000/v2/repository/index/ensemble/load

curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 100, "bad_words": "", "stop_words": ""}'
curl --location --request GET 'http://localhost:8000/v2/models/ensemble/stats'  
curl localhost:8000/api/status/

curl -X POST http://localhost:8000/v2/models/ensemble/generate_stream   -d '{"id": "42", "text_input": "<|image|>If I had to write a haiku for this one", "image_url_input": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png", "parameters": {"max_tokens": 16, "beam_width": 1, "end_id": 128001, "pad_id": 128004, "top_k": 1, "top_p": 0, "stream": false, "temperature": 0}}'

curl -X POST http://localhost:8000/v2/models/ensemble/generate_stream     -d '{"id": "42", "text_input": "<|image|>If I had to write a haiku for this one", "image_url": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png", "parameters": {"max_tokens": 16, "beam_width": 1, "end_id": 128001, "pad_id": 128004, "top_k": 1, "top_p": 0, "stream": false, "temperature": 0}}'





docker build -t llama_vision_triton .

docker run --gpus all -v /mnt/model_data:/model_data -v $(pwd):/notebooks -e HF_TOKEN=YOUR_HF_TOKEN -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 llama_vision_triton:latest jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root


docker run --gpus all -v /mnt/model_data:/model_data -v $(pwd):/notebooks -e HF_TOKEN=YOUR_HF_TOKEN -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 llama_vision_triton:latest


# curl -X POST http://localhost:8000/v2/models/ensemble/generate_stream     -d '{"id": "42", "text_input": "<|image|>If I had to write a haiku for this one", "parameters": {"max_tokens": 16, "beam_width": 1, "end_id": 128001, "pad_id": 128004, "top_k": 1, "top_p": 0, "stream": false, "temperature": 0}}'

curl -X POST localhost:8000/v2/models/ensemble/generate_stream \
-d '{"id": "42", "text_input": "<|image|>If I had to write a haiku for this one", "image_url_input": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png", "parameters": {"max_tokens": 16, "beam_width": 1, "end_id": 128001, "pad_id": 128004, "top_k": 1, "top_p": 0, "stream": false, "temperature": 0}}'
