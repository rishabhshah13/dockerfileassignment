#!/usr/bin/env python3
import os
import subprocess
import argparse
from transformers import AutoModelForVision2Seq, AutoTokenizer
from huggingface_hub import snapshot_download

def run_command(command, error_msg):
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{error_msg}: {result.stderr}")

def tie_model_weights(model_path, output_path):
    print("Tying model weights...")
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="auto")

    # Tie weights (for LLaMA, typically embedding and output layers)
    model.tie_weights()

    # Save the model with tied weights
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Model weights tied and saved successfully!")

def build_vision_engine(model_path, output_dir, max_batch_size=1, tp_size=2):
    # Tie weights first and use the new path
    tied_model_path = f"{output_dir}/tied_model"
    tie_model_weights(model_path, tied_model_path)
    
    cmd = (
        f"CUDA_VISIBLE_DEVICES=0,1 "  # Both GPUs
        f"MPI_LOCALNRANKS=2 "  # Force 2 ranks
        f"OMP_NUM_THREADS=1 "  # Limit CPU threads
        f"python3 /app/tensorrt_llm/examples/multimodal/build_visual_engine.py "
        f"--model_type mllama "
        f"--model_path {tied_model_path} "
        f"--output_dir {output_dir}/vision/ "
        f"--max_batch_size {max_batch_size}"
    )
    run_command(cmd, "Failed to build vision encoder engine")

def build_decoder_engine(model_path, output_dir, max_batch_size=1, tp_size=2):
    # Tie weights first and use the new path
    tied_model_path = f"{output_dir}/tied_model"
    tie_model_weights(model_path, tied_model_path)
    
    checkpoint_dir = f"{output_dir}/trt_ckpts"
    cmd1 = (
        f"CUDA_VISIBLE_DEVICES=0,1 "
        f"MPI_LOCALNRANKS=2 "
        f"OMP_NUM_THREADS=1 "
        f"python3 /app/tensorrt_llm/examples/mllama/convert_checkpoint.py "
        f"--model_dir {tied_model_path} "
        f"--output_dir {checkpoint_dir} "
        f"--dtype int4"
    )
    run_command(cmd1, "Failed to convert checkpoint")

    cmd2 = (
        f"CUDA_VISIBLE_DEVICES=0,1 "
        f"MPI_LOCALNRANKS=2 "
        f"OMP_NUM_THREADS=1 "
        f"trtllm-build "
        f"--checkpoint_dir {checkpoint_dir} "
        f"--output_dir {output_dir}/llm/ "
        f"--max_num_tokens 2048 "
        f"--max_seq_len 1024 "
        f"--workers {tp_size} "
        f"--gemm_plugin auto "
        f"--max_batch_size {max_batch_size} "
        f"--max_encoder_input_len 6404 "
        f"--input_timing_cache /app/tensorrt_llm/model.cache "
        f"--use_paged_context_fmha enable"  # Memory-efficient mode
    )
    run_command(cmd2, "Failed to build decoder engine")

def main():
    parser = argparse.ArgumentParser(description="Build MLLaMA TensorRT engines")
    parser.add_argument("--model_path", required=True, help="Path to LLaMA 3.2 11B Vision model")
    parser.add_argument("--output_dir", default="/model_engine", help="Output directory")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Max batch size")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallelism size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Building vision encoder...")
    build_vision_engine(args.model_path, args.output_dir, args.max_batch_size, args.tp_size)
    print("Building decoder...")
    build_decoder_engine(args.model_path, args.output_dir, args.max_batch_size, args.tp_size)
    print("Engines built successfully!")

if __name__ == "__main__":
    main()