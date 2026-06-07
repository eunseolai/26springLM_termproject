from transformers import AutoTokenizer
from src.constants import label2id

def get_preprocess_fn(tokenizer: AutoTokenizer):
        
    def preprocess(example):
        context_text = " ".join(example["context"]["contexts"])

        model_input = tokenizer(
            example["question"],
            context_text,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        model_input["labels"] = label2id[
            example["final_decision"]
            ]

        return model_input

    return preprocess