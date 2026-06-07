from src.constants import label2id, id2label

from transformers import AutoModelForSequenceClassification
def get_model(model_name, peft_method=None):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )