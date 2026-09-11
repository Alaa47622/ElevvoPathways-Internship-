import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="microsoft/phi-2")
    parser.add_argument(
        "--adapter-path",
        default="artifacts/adapters/phi2-samsum",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/merged_model",
    )
    args = parser.parse_args()

    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)

    print("Loading full-precision/FP16 base model for merge...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging adapter...")
    merged = model.merge_and_unload()

    merged.save_pretrained(output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output)

    print(f"Merged model saved to: {output}")


if __name__ == "__main__":
    main()
