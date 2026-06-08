from peft import PrefixTuningConfig, get_peft_model
from src.config import PT_DEFAULTS
from copy import deepcopy

def apply_prefix_tuning(model, num_virtual_tokens=None):

    pt_config = PT_DEFAULTS.deepcopy()

    if num_virtual_tokens is not None:
        pt_config["num_virtual_tokens"] = num_virtual_tokens

    peft_config = PrefixTuningConfig(**pt_config)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model