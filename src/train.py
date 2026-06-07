def main():
    # 1. dataset
    from datasets import load_dataset
    ds_a = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    ds_l = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

    def split_ds(dataset, **kwargs):
        return dataset.train_test_split(**kwargs)

    # 2. model and tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    label2id = {
    "yes": 0,
    "no": 1,
    "maybe": 2
    }

    id2label = {
        0: "yes",
        1: "no",
        2: "maybe"
    }

    def get_model(model_name):
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=id2label,
            label2id=label2id
        )
    
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    model = get_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. preprocessing
    def preprocess(example):
        context_text = " ".join(example["context"]["contexts"])

        model_input = tokenizer(
            example["question"],
            context_text,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        label = label2id[example["final_decision"]]
        model_input["labels"] = label

        return model_input
    
    processed_a = ds_a["train"].map(preprocess, batched=False)
    processed_a = processed_a.remove_columns(
        [    "pubid", "question", "context", "long_answer", "final_decision"]
        )
    
    processed_l = ds_l["train"].map(preprocess, batched=False)
    processed_l = processed_l.remove_columns(
        ["pubid", "question", "context", "long_answer", "final_decision"]
    )

    # 4. training
    from sklearn.utils.class_weight import compute_class_weight
    from collections import Counter
    from transformers import TrainingArguments, Trainer
    import torch
    import torch.nn as nn
    import numpy as np
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
    class_weights = torch.ones(len(label2id), dtype=torch.float).to(model.device)
    for label, idx in label_counts.items():
        count = label_counts.get(label, 0)
        if count > 0:
            class_weights[idx] = N / (K * count)


    # Define a custom Trainer to use class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")

            outputs = model(**inputs)
            logits = outputs.get("logits")

            device = logits.device

            # Compute the loss with class weights
            loss_fct = nn.CrossEntropyLoss(
                weight=class_weights.to(logits.device)
                )  # Adjust weights as needed
            loss = loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss
        
    def safe(arr, idx):
        return arr[idx] if idx < len(arr) else 0.0
    
    import numpy as np

    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support
    )

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

        preds = np.argmax(logits, axis=-1)

        # macro metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro"
        )

        acc = accuracy_score(labels, preds)

        # per-class metrics
        precision_cls, recall_cls, f1_cls, support = precision_recall_fscore_support(
            labels,
            preds,
            average=None,
            labels=[0,1,2],
            zero_division=0
        )

        return {
            # macro
            "accuracy": acc,
            "macro_f1": f1_macro,
            "macro_precision": precision_macro,
            "macro_recall": recall_macro,

            # per-class
            "yes_precision": safe(precision_cls, 0),
            "no_precision": safe(precision_cls, 1),
            "maybe_precision": safe(precision_cls, 2),

            "yes_recall": safe(recall_cls, 0),
            "no_recall": safe(recall_cls, 1),
            "maybe_recall": safe(recall_cls, 2),

            "yes_f1": safe(f1_cls, 0),
            "no_f1": safe(f1_cls, 1),
            "maybe_f1": safe(f1_cls, 2),

            # optional (데이터 분포 확인용)
            "support_yes": safe(support, 0),
            "support_no": safe(support, 1),
            "support_maybe": safe(support, 2),
        }

    batch_size=8

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