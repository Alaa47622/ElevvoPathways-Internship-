from pathlib import Path
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .prompts import build_prompt


def get_compute_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def make_bnb_config(config: dict) -> BitsAndBytesConfig:
    dtype_name = config["compute_dtype"]
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["quant_type"],
        bnb_4bit_use_double_quant=config["use_double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str, quantization: dict | None = None):
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if quantization and quantization.get("load_in_4bit", False):
        kwargs["quantization_config"] = make_bnb_config(quantization)
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_adapter_model(base_model_name: str, adapter_path: str, quantization: dict | None = None):
    base = load_base_model(base_model_name, quantization)
    return PeftModel.from_pretrained(base, adapter_path)


def prompt_for_generation(dialogue: str) -> str:
    return build_prompt(dialogue)


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_summary(model, tokenizer, dialogue: str, generation: dict) -> str:
    prompt = prompt_for_generation(dialogue)
    device = get_input_device(model)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = generation.get("temperature", 0.0) > 0

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation.get("max_new_tokens", 96),
            do_sample=do_sample,
            temperature=generation.get("temperature", 0.2) if do_sample else None,
            top_p=generation.get("top_p", 0.9) if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text.split("###")[0].strip()


def model_size_gb(model) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / (1024 ** 3)
