# Use an official NVIDIA Triton Server base image as the starting point
# Replace <xx.yy> with the desired version (e.g., 24.12 for a recent release as of Feb 2025)
ARG TRITON_VERSION=24.12
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3 AS base

# Set working directory
WORKDIR /opt/triton

# Install additional dependencies (if needed for your custom backend)
# For example, Python dependencies or tools for building custom backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install numpy grpcio-tools

# Example: Copy a custom Python backend into the container
# Replace './my_custom_backend' with the path to your backend code
COPY ./my_custom_backend /opt/tritonserver/backends/my_custom_backend

# If your custom backend requires compilation (e.g., C++ backend), add build steps here
# Example for a C++ backend (uncomment and adjust as needed):
# RUN mkdir -p /opt/tritonserver/backends/my_custom_backend/build && \
#     cd /opt/tritonserver/backends/my_custom_backend/build && \
#     cmake .. -DTRITON_ENABLE_GPU=ON && \
#     make install

# Set environment variables for Triton
ENV TRITON_SERVER_PATH=/opt/tritonserver/bin/tritonserver
ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH

# Copy model repository (optional, adjust path to your model repository)
# This assumes you have a model repository in './model_repository'
COPY ./model_repository /models

# Expose default Triton ports: HTTP (8000), gRPC (8001), Metrics (8002)
EXPOSE 8000 8001 8002

# Start Triton Server with the model repository
CMD ["tritonserver", "--model-repository=/models"]