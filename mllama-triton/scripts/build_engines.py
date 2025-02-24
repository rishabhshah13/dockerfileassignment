#!/usr/bin/env python3
import os
import subprocess
import argparse

def run_command(command, error_msg):
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{error_msg}: {result.stderr}")

def build_vision_engine(model_path, output_dir, max_batch_size, tp_size, quantization):
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
    cmd1 = (
        f"CUDA_VISIBLE_DEVICES=0,1 "
        f"MPI_LOCALNRANKS=2 "
        f"OMP_NUM_THREADS=1 "
        f"python3 /app/tensorrt_llm/examples/mllama/convert_checkpoint.py "
        f"--model_dir {model_path} "
        f"--output_dir {checkpoint_dir} "
        f"--dtype {quantization}"
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
        f"--use_paged_context_fmha enable"
    )
    run_command(cmd2, "Failed to build decoder engine")

def main():
    parser = argparse.ArgumentParser(description="Build MLLaMA TensorRT engines")
    parser.add_argument("--model_path", required=True, help="Path to LLaMA 3.2 11B Vision model")
    parser.add_argument("--output_dir", default="/model_engine", help="Output directory")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Max batch size")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallelism size")
    parser.add_argument("--quantization", default="int4", help="Quantization type (int4, fp8)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Building vision encoder...")
    build_vision_engine(args.model_path, args.output_dir, args.max_batch_size, args.tp_size, args.quantization)
    print("Building decoder...")
    build_decoder_engine(args.model_path, args.output_dir, args.max_batch_size, args.tp_size, args.quantization)
    print("Engines built successfully!")

if __name__ == "__main__":
    main()
    
    # #!/usr/bin/env python3
# import os
# import subprocess
# import argparse

# def run_command(command, error_msg):
#     result = subprocess.run(command, shell=True, check=False)
#     if result.returncode != 0:
#         raise RuntimeError(f"{error_msg}: {result.stderr}")

# def build_vision_engine(model_path, output_dir, max_batch_size=1, tp_size=2):
#     # cmd = (
#     #     f"CUDA_VISIBLE_DEVICES=0,1 "  # Both GPUs
#     #     f"MPI_LOCALNRANKS=2 "  # Force 2 ranks for multi-GPU
#     #     f"python3 /app/tensorrt_llm/examples/multimodal/build_visual_engine.py "
#     #     f"--model_type mllama "
#     #     f"--model_path {model_path} "
#     #     f"--output_dir {output_dir}/vision/ "
#     #     f"--max_batch_size {max_batch_size} "
#     #     f"--tp_size {tp_size} "
#     #     f"--world_size {tp_size} "  # Explicit world size
#     #     f"--dtype fp16"
#     # )
#     cmd = (
#         f"CUDA_VISIBLE_DEVICES=0,1 "
#         f"MPI_LOCALNRANKS=2 "
#         f"python3 /app/tensorrt_llm/examples/multimodal/build_visual_engine.py "
#         f"--model_type mllama "
#         f"--model_path {model_path} "
#         f"--output_dir {output_dir}/vision/ "
#         f"--max_batch_size {max_batch_size} "
#         f"--tp_size {tp_size} "
#         f"--dtype int4"
#     )
#     run_command(cmd, "Failed to build vision encoder engine")

# def build_decoder_engine(model_path, output_dir, max_batch_size=1, tp_size=2):
#     checkpoint_dir = f"{output_dir}/trt_ckpts"
#     # cmd1 = (
#     #     f"CUDA_VISIBLE_DEVICES=0,1 "
#     #     f"MPI_LOCALNRANKS=2 "
#     #     f"python3 /app/tensorrt_llm/examples/mllama/convert_checkpoint.py "
#     #     f"--model_dir {model_path} "
#     #     f"--output_dir {checkpoint_dir} "
#     #     f"--dtype fp16"
#     # )
#     cmd1 = (
#         f"CUDA_VISIBLE_DEVICES=0,1 "
#         f"MPI_LOCALNRANKS=2 "
#         f"python3 /app/tensorrt_llm/examples/mllama/convert_checkpoint.py "
#         f"--model_dir {model_path} "
#         f"--output_dir {checkpoint_dir} "
#         f"--dtype int4"
#     )
#     run_command(cmd1, "Failed to convert checkpoint")

#     cmd2 = (
#         f"CUDA_VISIBLE_DEVICES=0,1 "
#         f"MPI_LOCALNRANKS=2 "
#         f"trtllm-build "
#         f"--checkpoint_dir {checkpoint_dir} "
#         f"--output_dir {output_dir}/llm/ "
#         f"--max_num_tokens 2048 "
#         f"--max_seq_len 1024 "
#         f"--workers {tp_size} "
#         f"--gemm_plugin auto "
#         f"--max_batch_size {max_batch_size} "
#         f"--max_encoder_input_len 6404 "
#         f"--input_timing_cache /app/tensorrt_llm/model.cache"
#     )
#     run_command(cmd2, "Failed to build decoder engine")

# def main():
#     parser = argparse.ArgumentParser(description="Build MLLaMA TensorRT engines")
#     parser.add_argument("--model_path", required=True, help="Path to LLaMA 3.2 11B Vision model")
#     parser.add_argument("--output_dir", default="/model_engine", help="Output directory")
#     parser.add_argument("--max_batch_size", type=int, default=1, help="Max batch size")
#     parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallelism size")
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
#     print("Building vision encoder...")
#     build_vision_engine(args.model_path, args.output_dir, args.max_batch_size, args.tp_size)
#     print("Building decoder...")
#     build_decoder_engine(args.model_path, args.output_dir, args.max_batch_size, args.tp_size)
#     print("Engines built successfully!")

# if __name__ == "__main__":
#     main()