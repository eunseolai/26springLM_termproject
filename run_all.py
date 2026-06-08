from src.peft.factory import build_model
from src.data.dataset import load_pubmedqa_datasets, split_ds
from src.data.preprocess import get_preprocess_fn
from src.training.trainer import WeightedTrainer
from src.utils.metrics import compute_metrics
from transformers import AutoTokenizer, TrainingArguments
from collections import Counter
import torch
from train import run_training

MODEL_LIST = [
    "bert-base-uncased",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
]

PEFT_LIST = [
    "none",
    "lora",
    "prefix",
    "ia3",
    "dora",
    "adapter"
]

import subprocess

methods = ["lora", "dora", "ia3", "prefix", "adapter"]

for method in methods:
    subprocess.run([
        "python", "train.py",
        "--tuning_method", method,
        "--use_phase1",
        "--use_phase2"
    ])