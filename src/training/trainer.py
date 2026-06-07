import torch.nn as nn
from transformers import Trainer

class WeightedTrainer(Trainer):

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        device = logits.device

        # Compute the loss with class weights
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
            )  # Adjust weights as needed
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss