import os
import csv
from datetime import datetime

from collections import Counter
import time
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from src.data.dataset import split_ds
from src.data.preprocess import get_preprocess_fn
from src.training.trainer import WeightedTrainer
from src.utils.metrics import compute_metrics
from src.constants import label2id

def run_two_phase_training(model, tokenizer, ds_a, ds_l, args):
    if args.method == "prefix":
        max_length = 512 - args.prefix_num_virtual_tokens
    else:
        max_length = 512
    
    # 1. preprocessing
    preprocess = get_preprocess_fn(
        tokenizer, max_length=max_length
    )
    processed_a = ds_a["train"].map(preprocess, batched=False)
    processed_a = processed_a.remove_columns(
        ["pubid", "question", "context", "long_answer", "final_decision"]
    )
    processed_l = ds_l["train"].map(preprocess, batched=False)
    processed_l = processed_l.remove_columns(
        ["pubid", "question", "context", "long_answer", "final_decision"]
    )

    # 2. split
    split_l = split_ds(processed_l, test_size=0.2, seed=args.seed)

    X_train_a = processed_a
    X_train_l = split_l["train"]
    X_test_l = split_l["test"]

    # 3. class weights for phase 1
    label_counts = Counter(ds_a["train"]["final_decision"])
    N = sum(label_counts.values())
    K = len(label2id)
    class_weights = torch.ones(len(label2id), dtype=torch.float)

    for label, idx in label2id.items():
        count = label_counts.get(label, 0)
        if count > 0:
            class_weights[idx] = N / (K * count)

    batch_size = args.batch_size
    safe_model_name = args.model_name.split("/")[-1]
    total_start_time = time.perf_counter()
    load_best = False if args.method == "prefix" else True

    # 4. phase 1
    args_stage1 = TrainingArguments(
        output_dir=f"./outputs/{safe_model_name}/{args.method}/checkpoint_stage1",
        num_train_epochs=args.phase1_epochs,
        learning_rate=args.phase1_lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=max(1, len(X_train_a) // batch_size // 10),
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        fp16=True,
        dataloader_num_workers=2,
        label_names=["labels"],
    )

    trainer_stage1 = WeightedTrainer(
        model=model,
        args=args_stage1,
        train_dataset=X_train_a,
        eval_dataset=X_test_l,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ],
    )

    stage1_start_time = time.perf_counter()
    trainer_stage1.train()
    stage1_time = time.perf_counter() - stage1_start_time
    trainer_stage1.save_model()

    metrics_stage1 = trainer_stage1.evaluate()
    print("Stage 1 metrics:", metrics_stage1)
    print(f"Stage 1 training time: {stage1_time:.2f} seconds")

    # 5. phase 2
    num_train_epochs = args.phase2_epochs
    total_steps = max(1, len(X_train_l) // batch_size * num_train_epochs)

    args_stage2 = TrainingArguments(
        output_dir=f"./outputs/{safe_model_name}/{args.method}/checkpoint_stage2",
        num_train_epochs=num_train_epochs,
        learning_rate=args.phase2_lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        fp16=True,
        dataloader_num_workers=2,
        warmup_steps=int(0.1 * total_steps),
        weight_decay=0.01,
        label_names=["labels"],
    )

    trainer_stage2 = Trainer(
        model=model,
        args=args_stage2,
        train_dataset=X_train_l,
        eval_dataset=X_test_l,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ],
    )

    stage2_start_time = time.perf_counter()
    trainer_stage2.train()
    stage2_time = time.perf_counter() - stage2_start_time
    trainer_stage2.save_model()

    metrics_stage2 = trainer_stage2.evaluate()
    print(f"Stage 2 training time: {stage2_time:.2f} seconds")

    total_time = time.perf_counter() - total_start_time
    print(f"Total training time: {total_time:.2f} seconds")

    metrics_stage2["stage1_eval_loss"] = metrics_stage1["eval_loss"]
    metrics_stage2["stage2_eval_loss"] = metrics_stage2["eval_loss"]


    metrics_stage2["stage1_time_sec"] = stage1_time
    metrics_stage2["stage2_time_sec"] = stage2_time
    metrics_stage2["total_time_sec"] = total_time
    
    save_metrics_to_csv(metrics_stage2, args)

    print("Final metrics:", metrics_stage2)
    return metrics_stage2

def save_metrics_to_csv(metrics, args, path="results/experiment_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": args.model_name,
        "method": args.method,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "batch_size": args.batch_size,
        "phase1_lr": args.phase1_lr,
        "phase2_lr": args.phase2_lr,
        "safe_model_name": args.model_name.split("/")[-1],
    }

    row.update(metrics)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)