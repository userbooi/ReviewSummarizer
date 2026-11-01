from datasets import load_from_disk
from prepare_dataset import save_dataset
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorWithPadding
import torch

checkpoint = "allenai/PRIMERA-multinews"
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "amasum_hf"
if not DATA_DIR.exists():
    RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "amasum"
    save_dataset(DATA_DIR, RAW_DATA_PATH)

ds = load_from_disk(DATA_DIR)
# print(ds)
print(ds["train"][0])


def tokenize_function(example):
    source = " <doc-sep> ".join(example["documents"])
