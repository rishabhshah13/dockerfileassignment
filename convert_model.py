import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse

def convert_model(args):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Export model for TensorRT-LLM
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    convert_model(args)
