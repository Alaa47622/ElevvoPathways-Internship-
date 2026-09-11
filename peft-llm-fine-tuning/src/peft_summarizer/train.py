import argparse
import json
import os
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import load_yaml
from .prompts import build_prompt, build_training_text


def build_bnb_config(qcfg):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qcfg["quant_type"],
        bnb_4bit_use_double_quant=qcfg["use_double_quant"],
        bnb_4bit_compute_dtype=torch.float16,
    )


def tokenize_example(example, tokenizer, max_length):
    prompt = build_prompt(example["dialogue"])
    full_text = build_training_text(example["dialogue"], example["summary"])

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    full = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = min(len(prompt_ids), max_length)
    labels = input_ids.copy()

    # Do not train on the prompt tokens; train only on the reference summary.
    labels[:prompt_len] = [-100] * prompt_len

    # Ignore padding.
    labels = [
        token if mask else -100
        for token, mask in zip(labels, attention_mask)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["training"]["seed"])

    model_name = cfg["model"]["name"]
    dataset_name = cfg["dataset"]["name"]
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training is configured for CUDA. Use a Colab T4/Linux NVIDIA GPU."
        )

    quant_cfg = cfg["quantization"]
    bnb_config = build_bnb_config(quant_cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["qlora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )

    # Apply LoRA directly so the Trainer saves a PEFT adapter.
    from peft import get_peft_model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ds = load_dataset(dataset_name)
    train_ds = ds[cfg["dataset"]["train_split"]]
    eval_ds = ds[cfg["dataset"]["eval_split"]]

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    max_len = cfg["training"]["max_seq_length"]

    train_ds = train_ds.map(
        lambda x: tokenize_example(x, tokenizer, max_len),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train set",
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_example(x, tokenizer, max_len),
        remove_columns=eval_ds.column_names,
        desc="Tokenizing validation set",
    )

    tcfg = cfg["training"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tcfg["num_train_epochs"],
        learning_rate=tcfg["learning_rate"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        fp16=tcfg["fp16"],
        bf16=tcfg["bf16"],
        report_to=[],
        remove_unused_columns=False,
        seed=tcfg["seed"],
    )

    mlcfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlcfg["tracking_uri"])
    mlflow.set_experiment(mlcfg["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({
            "model": model_name,
            "dataset": dataset_name,
            "epochs": tcfg["num_train_epochs"],
            "learning_rate": tcfg["learning_rate"],
            "batch_size": tcfg["per_device_train_batch_size"],
            "gradient_accumulation_steps": tcfg["gradient_accumulation_steps"],
            "max_seq_length": max_len,
            "lora_r": lora_cfg["r"],
            "lora_alpha": lora_cfg["alpha"],
            "lora_dropout": lora_cfg["dropout"],
            "quantization": "4-bit NF4",
        })

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )

        result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(metrics)

        metadata = {
            "model": model_name,
            "dataset": dataset_name,
            "config": cfg,
            "train_metrics": metrics,
        }
        with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(str(output_dir / "training_metadata.json"))

    print(f"\nAdapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
