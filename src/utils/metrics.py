from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

import numpy as np

def safe(arr, idx):
    return arr[idx] if idx < len(arr) else 0.0

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=-1)

    # macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
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

        # 데이터 분포 확인용
        "support_yes": safe(support, 0),
        "support_no": safe(support, 1),
        "support_maybe": safe(support, 2),
    }