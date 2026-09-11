import argparse

from .config import load_yaml
from .model_utils import generate_summary, load_adapter_model, load_base_model, load_tokenizer


def load_for_inference(config_path="configs/inference.yaml"):
    cfg = load_yaml(config_path)
    model_name = cfg["model"]["name"]
    adapter_path = cfg["model"].get("adapter_path")

    tokenizer = load_tokenizer(model_name)

    if adapter_path:
        model = load_adapter_model(
            model_name,
            adapter_path,
            {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "use_double_quant": True,
                "compute_dtype": "float16",
            },
        )
    else:
        model = load_base_model(model_name)

    return model, tokenizer, cfg["generation"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--dialogue", required=True)
    args = parser.parse_args()

    model, tokenizer, generation = load_for_inference(args.config)
    print(generate_summary(model, tokenizer, args.dialogue, generation))


if __name__ == "__main__":
    main()
