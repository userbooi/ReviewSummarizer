from datasets import load_dataset

def save_dataset():
    ds = load_dataset("talhaa/nlp-google-reviews-dataset")
    ds.save_to_disk("../datasets/google-reviews-dataset")
