import argparse
import json
from pathlib import Path

import evaluate as hf_evaluate
from datasets import load_dataset

from .config import load_yaml
from .model_utils import generate_summary, load_adapter_model, load_base_model, load_tokenizer


def compute_rouge(predictions, references):
    rouge = hf_evaluate.load("rouge")
    result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {k: float(v) for k, v in result.items()}


def run_evaluation(model, tokenizer, ds, generation):
    predictions = []
    references = []

    for i, row in enumerate(ds):
        pred = generate_summary(model, tokenizer, row["dialogue"], generation)
        predictions.append(pred)
        references.append(row["summary"])

        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(ds)}")

    return predictions, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--compare-base", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg["model"]["name"]

    ds = load_dataset(
        cfg["dataset"]["name"],
        split=cfg["dataset"]["test_split"],
    )
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tokenizer = load_tokenizer(model_name)

    if not args.adapter_path:
        model = load_base_model(model_name, cfg["quantization"])
        name = "base"
    else:
        model = load_adapter_model(
            model_name,
            args.adapter_path,
            cfg["quantization"],
        )
        name = "finetuned"

    predictions, references = run_evaluation(
        model,
        tokenizer,
        ds,
        cfg["generation"],
    )

    metrics = compute_rouge(predictions, references)

    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)

    with (output / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (output / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, pred in zip(ds, predictions):
            f.write(json.dumps({
                "id": row["id"],
                "dialogue": row["dialogue"],
                "reference": row["summary"],
                "prediction": pred,
            }, ensure_ascii=False) + "\n")

    print(f"\n{name} ROUGE:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if args.compare_base and args.adapter_path:
        base_file = output / "base.json"
        if not base_file.exists():
            print("\nBase results not found; run baseline first.")
            return

        with base_file.open("r", encoding="utf-8") as f:
            base = json.load(f)

        print("\nComparison:")
        print(f"{'Metric':<12} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
        print("-" * 48)

        for metric in metrics:
            b = base[metric]
            ft = metrics[metric]
            print(f"{metric:<12} {b:>10.4f} {ft:>12.4f} {ft-b:>+10.4f}")


if __name__ == "__main__":
    main()
