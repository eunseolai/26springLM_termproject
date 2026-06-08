from peft import LoraConfig, get_peft_model
from src.config import LORA_DEFAULTS

def apply_lora(model, r=None):

    lora_config = LORA_DEFAULTS.copy()

    if r is not None:
        lora_config["r"] = r

    peft_config = LoraConfig(**lora_config)

    model.print_trainable_parameters()

    return get_peft_model(model, peft_config)