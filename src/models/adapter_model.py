from adapters import AutoAdapterModel

def get_adapter_model(model_name, num_labels=3):
    model = AutoAdapterModel.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    model.add_adapter("pubmedqa_adapter")
    model.add_classification_head("pubmedqa_adapter", num_labels=num_labels)

    model.train_adapter("pubmedqa_adapter")
    model.set_active_adapters("pubmedqa_adapter")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || "
          f"all params: {total_params} || "
          f"trainable%: {100 * trainable_params / total_params:.4f}%"
          )

    return model