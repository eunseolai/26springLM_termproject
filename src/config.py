LORA_DEFAULTS = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["query", "value"],
    "bias": "none",
    "task_type": "SEQ_CLS"
}

PT_DEFAULTS = {
    "task_type": "SEQ_CLS",
    "num_virtual_tokens": 20,
    "prefix_projection": True
    # project the prefix tokens to a higher dimension: stability
}

IA3_DEFAULTS = {
    "target_modules": ["query", "value", "dense"],
    "feedforward_modules": ["dense"],
    "task_type": "SEQ_CLS"
}