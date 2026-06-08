from peft import IA3Config, get_peft_model
from src.config import IA3_DEFAULTS

def apply_ia3(model):
    peft_config = IA3Config(**IA3_DEFAULTS)
    model.print_trainable_parameters()
    return get_peft_model(model, peft_config)