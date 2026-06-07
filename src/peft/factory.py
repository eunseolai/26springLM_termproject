# factory.py

from src.models.backbone import get_model as get_base_model
from src.models.adapter_model import get_adapter_model

from src.peft.lora import apply_lora
from src.peft.dora import apply_dora
from src.peft.ia3 import apply_ia3
from src.peft.prefix_tuning import apply_prefix_tuning


def build_model(model_name, method=None):

    # 1. Adapter 계열 (완전히 다른 모델)
    if method == "adapter":
        return get_adapter_model(model_name)

    # 2. 기본 backbone
    model = get_base_model(model_name)

    # 3. PEFT 계열
    if method == "lora":
        return apply_lora(model)

    elif method == "dora":
        return apply_dora(model)

    elif method == "prefix_tuning":
        return apply_prefix_tuning(model)

    elif method == "ia3":
        return apply_ia3(model)

    # 4. no PEFT
    else:
        return model