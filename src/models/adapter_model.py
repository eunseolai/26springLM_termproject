from adapters import AutoAdapterModel
from adapters.composition import Stack

def get_adapter_model(model_name, num_labels=3):
    adapter_name = "pubmedqa_adapter"
    head_name = "pubmedqa"

    model = AutoAdapterModel.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    model.add_adapter(adapter_name)
    model.add_classification_head(head_name, num_labels=num_labels)

    model.train_adapter(adapter_name)

    model.active_adapters = Stack(adapter_name)
    model.active_head = head_name

    print("Active adapters:", model.active_adapters)
    print("Active head:", model.active_head)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable_params} || "
        f"all params: {total_params} || "
        f"trainable%: {100 * trainable_params / total_params:.4f}%"
    )

    return model