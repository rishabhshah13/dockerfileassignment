#!/usr/bin/env python3
import os
import subprocess
import argparse

def run_command(command, error_msg):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{error_msg}: {result.returncode}")

def build_vision_engine(model_path, output_dir, max_batch_size, tp_size, quantization):
    os.makedirs(f"{output_dir}/vision/", exist_ok=True)
    cmd = (
        f"CUDA_VISIBLE_DEVICES=0,1 "
        f"MPI_LOCALNRANKS=2 "
        f"OMP_NUM_THREADS=1 "
        f"python3 /app/tensorrt_llm/examples/multimodal/build_visual_engine.py "
        f"--model_type mllama "
        f"--model_path {model_path} "
        f"--output_dir {output_dir}/vision/ "
        f"--max_batch_size {max_batch_size}"
    )
    run_command(cmd, "Failed to build vision encoder engine")

def build_decoder_engine(model_path, output_dir, max_batch_size, tp_size, quantization):
    checkpoint_dir = f"{output_dir}/trt_ckpts"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # MLLaMA requires bfloat16
    cmd1 = (
        f"CUDA_VISIBLE_DEVICES=0,1 "
        f"MPI_LOCALNRANKS=2 "
        f"OMP_NUM_THREADS=1 "
        f"python3 /app/tensorrt_llm/examples/mllama/convert_checkpoint.py "
        f"--model_dir {model_path} "
        f"--output_dir {checkpoint_dir} "
        f"--dtype float16"  # MLLaMA requires bfloat16
    )
    run_command(cmd1, "Failed to convert checkpoint")

    os.makedirs(f"{output_dir}/llm/", exist_ok=True)
    cmd2 = (
        f"CUDA_VISIBLE_DEVICES=0,1 "
        f"trtllm-build "
        f"--checkpoint_dir {checkpoint_dir} "
        f"--output_dir {output_dir}/llm/ "
        f"--gemm_plugin auto "
        f"--max_batch_size {max_batch_size} "
        f"--max_seq_len 2048 "
        f"--max_num_tokens 4096 "
        f"--max_encoder_input_len 6404 "
        f"--workers {tp_size}"
    )
    run_command(cmd2, "Failed to build decoder engine")

def main():
    parser = argparse.ArgumentParser(description="Build MLLaMA TensorRT engines")
    parser.add_argument("--model_path", required=True, help="Path to LLaMA 3.2 11B Vision model")
    parser.add_argument("--output_dir", default="/model_engine", help="Output directory")
    parser.add_argument("--max_batch_size", type=int, default=8, help="Max batch size")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallelism size")
    parser.add_argument("--quantization", default="none", help="Quantization type (not used for MLLaMA)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Building vision encoder...")
    build_vision_engine(args.model_path, args.output_dir, args.max_batch_size, args.tp_size, args.quantization)
    print("Building decoder...")
    build_decoder_engine(args.model_path, args.output_dir, args.max_batch_size, args.tp_size, args.quantization)
    print("Engines built successfully!")

if __name__ == "__main__":
    main()
