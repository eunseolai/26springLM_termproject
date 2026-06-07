from src.data.dataset import load_pubmedqa_datasets
from src.data.dataset import split_ds
from src.data.preprocess import get_preprocess_fn
from src.models.backbone import get_model
from src.training.trainer import WeightedTrainer
from src.utils.metrics import safe, compute_metrics
from src.constants import label2id, id2label

def main():
    # 1. dataset
    from datasets import load_dataset
    ds_a, ds_l = load_pubmedqa_datasets()

    # 2. model and tokenizer
    from transformers import AutoTokenizer
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    model = get_model(model_name, peft_method=None)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. preprocessing
    preprocess = get_preprocess_fn(tokenizer)

    processed_a = ds_a["train"].map(preprocess, batched=False)
    processed_a = processed_a.remove_columns(
        ["pubid", "question", "context", "long_answer", "final_decision"]
        )
    
    processed_l = ds_l["train"].map(preprocess, batched=False)
    processed_l = processed_l.remove_columns(
        ["pubid", "question", "context", "long_answer", "final_decision"]
    )

    # 4. training
    from collections import Counter
    from transformers import TrainingArguments, Trainer
    import torch
    import torch.nn as nn
    from transformers import EarlyStoppingCallback

    # train_ds, eval_ds = split_ds(processed_a, test_size=0.2, seed=42).values()
    ds_a = split_ds(processed_a, test_size=0.2, seed=42)
    ds_l = split_ds(processed_l, test_size=0.2, seed=42)
    X_train_a, X_test_a = ds_a["train"], ds_a["test"]
    X_train_l, X_test_l = ds_l["train"], ds_l["test"]

    # Compute class weights
    ds = ds_a
    label_counts = Counter(ds_a["train"]["final_decision"])

    N = sum(label_counts.values())
    K = len(label2id)

    # Initialize class weights to 1.0 for all classes
    class_weights = torch.ones(len(label2id), dtype=torch.float)
    for label, idx in label_counts.items():
        count = label_counts.get(label, 0)
        if count > 0:
            class_weights[idx] = N / (K * count)


    batch_size=8
    # phase 1
    args_stage1 = TrainingArguments(
        output_dir="./checkpoint",
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        # small steps for small dataset
        logging_steps = max(1, len(X_train_a)//batch_size//10),
        load_best_model_at_end=True,
        seed=42,
        fp16=True,  # 16bit->memory efficient
        dataloader_num_workers=2  # 병렬 load
    )

    # Initialize Trainer 1
    trainer_stage1 = WeightedTrainer(
        model=model,
        args=args_stage1,
        train_dataset=X_train_a,
        eval_dataset=X_test_a,
        compute_metrics=compute_metrics,
        class_weights=class_weights,

        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    # phase 1 training
    trainer_stage1.train()
    trainer_stage1.save_model()
    metrics = trainer_stage1.evaluate()
    print(metrics)


    # phase 2
    num_train_epochs=8
    total_steps = len(X_train_l) // batch_size * num_train_epochs

    args_stage2 = TrainingArguments(
        output_dir="./checkpoint",
        num_train_epochs=num_train_epochs, # early stopping -> ok
        learning_rate=5e-6, # smaller lr for fine adjustment

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        seed=42,
        fp16=True,  # 16bit->memory efficient
        dataloader_num_workers=2,  # 병렬 load
        warmup_steps = int(0.1 * total_steps), # 안정화
        weight_decay=0.01  # prevents overfitting
    )

    # Initialize Trainer 2
    trainer_stage2 = Trainer(
        model=model,
        args=args_stage2,
        train_dataset=X_train_l,
        eval_dataset=X_test_l,
        compute_metrics=compute_metrics,   # 그대로 사용

        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    # phase 2 training: no class weights, smaller lr, more epochs
    trainer_stage2.train()
    trainer_stage2.save_model()
    metrics = trainer_stage2.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()