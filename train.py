from transformers import AutoTokenizer

from src.data.dataset import load_pubmedqa_datasets
from src.peft.factory import build_model
from src.training.train_pipeline import run_two_phase_training


def main():
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

    method = "lora"  # argparse로 바꿀 예정

    # 1. dataset
    ds_a, ds_l = load_pubmedqa_datasets()

    # 2. model & tokenizer
    model = build_model(model_name=model_name, method=method)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. training
    run_two_phase_training(
        model=model,
        tokenizer=tokenizer,
        ds_a=ds_a,
        ds_l=ds_l,
        method=method,
    )


if __name__ == "__main__":
    main()