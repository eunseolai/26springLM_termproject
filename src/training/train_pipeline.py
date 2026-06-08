from collections import Counter

import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from src.data.dataset import split_ds
from src.data.preprocess import get_preprocess_fn
from src.training.trainer import WeightedTrainer
from src.utils.metrics import compute_metrics
from src.constants import label2id


def run_two_phase_training(model, tokenizer, ds_a, ds_l, method):
    # 1. preprocessing
    preprocess = get_preprocess_fn(tokenizer)

    processed_a = ds_a["train"].map(preprocess, batched=False)
    processed_a = processed_a.remove_columns(
        ["pubid", "question", "context", "long_answer", "final_decision"]
    )

    processed_l = ds_l["train"].map(preprocess, batched=False)
    processed_l = processed_l.remove_columns(
        ["pubid", "question", "context", "long_answer", "final_decision"]
    )

    # 2. split
    split_l = split_ds(processed_l, test_size=0.2, seed=42)

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

    batch_size = 8

    # 4. phase 1
    args_stage1 = TrainingArguments(
        output_dir=f"./outputs/{method}/checkpoint_stage1",
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=max(1, len(X_train_a) // batch_size // 10),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        fp16=True,
        dataloader_num_workers=2,
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

    trainer_stage1.train()
    trainer_stage1.save_model()

    metrics_stage1 = trainer_stage1.evaluate()
    print("Stage 1 metrics:", metrics_stage1)

    # 5. phase 2
    num_train_epochs = 8
    total_steps = len(X_train_l) // batch_size * num_train_epochs

    args_stage2 = TrainingArguments(
        output_dir=f"./outputs/{method}/checkpoint_stage2",
        num_train_epochs=num_train_epochs,
        learning_rate=5e-6,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        fp16=True,
        dataloader_num_workers=2,
        warmup_steps=int(0.1 * total_steps),
        weight_decay=0.01,
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

    trainer_stage2.train()
    trainer_stage2.save_model()

    metrics_stage2 = trainer_stage2.evaluate()
    print("Stage 2 metrics:", metrics_stage2)

    return metrics_stage2