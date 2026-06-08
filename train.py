import argparse
from transformers import AutoTokenizer

from src.data.dataset import load_pubmedqa_datasets
from src.peft.factory import build_model
from src.training.train_pipeline import run_two_phase_training

def parse_args():
    parser = argparse.ArgumentParser(description="Two-phase training for PubMedQA")
    
    parser.add_argument(
        "--model_name", 
        type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", 
        help="Pre-trained model name"
    )

    parser.add_argument(
        "--method", 
        type=str, required=True, default="lora", 
        choices=["none", "lora", "prefix", "ia3", "dora", "adapter"],
        help="PEFT method to use"
        )
    
    parser.add_argument(
        "--batch_size",
        type=int, default=8
    )

    parser.add_argument(
        "--phase1_epochs",
        type=int, default=2
    )

    parser.add_argument(
        "--phase2_epochs",
        type=int, default=8
    )

    parser.add_argument(
        "--phase1_lr",
        type=float, default=2e-5
    )

    parser.add_argument(
        "--phase2_lr",
        type=float, default=5e-6
    )

    parser.add_argument(
        "--seed",
        type=int, default=42
    )

    # peft hyperparameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--dora_r", type=int, default=8)
    parser.add_argument("--prefix_num_virtual_tokens", type=int, default=20)
    parser.add_argument("--adapter_reduction_factor", type=int, default=16)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. dataset
    ds_a, ds_l = load_pubmedqa_datasets()

    # 2. model & tokenizer
    model = build_model(model_name=args.model_name, method=args.method)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 3. training
    run_two_phase_training(
        model=model,
        tokenizer=tokenizer,
        ds_a=ds_a,
        ds_l=ds_l,
        args=args
    )


if __name__ == "__main__":
    main()