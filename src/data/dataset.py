from datasets import load_dataset
def load_pubmedqa_datasets():
    ds_a = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    ds_l = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    return ds_a, ds_l

def split_ds(dataset, **kwargs):
    return dataset.train_test_split(**kwargs)
