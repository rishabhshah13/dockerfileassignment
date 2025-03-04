# Runtime stage
FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 AS runtime

# Set NVIDIA runtime for GPU access
ENV NVIDIA_VISIBLE_DEVICES=0,1  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]