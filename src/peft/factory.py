from src.models.backbone import get_model as get_base_model


def build_model(model_name, method=None, args=None):

    # adapter
    if method == "adapter":
        from src.models.adapter_model import get_adapter_model
        return get_adapter_model(model_name)

    # backbone
    model = get_base_model(model_name)

    # PEFT
    if method == "lora":
        from src.peft.lora import apply_lora
        return apply_lora(model)

    elif method == "dora":
        from src.peft.dora import apply_dora
        return apply_dora(model)

    elif method in ["prefix", "prefix_tuning"]:
        from src.peft.prefix_tuning import apply_prefix_tuning
        return apply_prefix_tuning(
            model,
            num_virtual_tokens=args.prefix_num_virtual_tokens
            )

    elif method == "ia3":
        from src.peft.ia3 import apply_ia3
        return apply_ia3(model)

    else:
        return model