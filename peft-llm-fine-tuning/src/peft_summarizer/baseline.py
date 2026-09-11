import argparse
import json
from pathlib import Path

import mlflow
from datasets import load_dataset

from .config import load_yaml
from .model_utils import generate_summary, load_base_model, load_tokenizer
from .evaluate import compute_rouge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg["model"]["name"]

    print(f"Loading base model: {model_name}")
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, cfg["quantization"])

    ds = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["test_split"])
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    predictions = []
    references = []

    for i, row in enumerate(ds):
        pred = generate_summary(model, tokenizer, row["dialogue"], cfg["generation"])
        predictions.append(pred)
        references.append(row["summary"])
        print(f"[{i+1}/{len(ds)}] {pred}")

    metrics = compute_rouge(predictions, references)

    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)

    with (output / "base.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (output / "base_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, pred in zip(ds, predictions):
            f.write(json.dumps({
                "id": row["id"],
                "dialogue": row["dialogue"],
                "reference": row["summary"],
                "prediction": pred,
            }, ensure_ascii=False) + "\n")

    print("\nBase model ROUGE:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
