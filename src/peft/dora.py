from peft import LoraConfig, get_peft_model
from src.config import LORA_DEFAULTS

def apply_dora(model, r=None):

    dora_config = LORA_DEFAULTS.copy()
    dora_config["use_dora"] = True
    
    if r is not None:
        dora_config["r"] = r

    peft_config = LoraConfig(**dora_config)
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model