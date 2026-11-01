from datasets import load_from_disk
from prepare_dataset import save_dataset
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
import torch

checkpoint = "allenai/PRIMERA-multinews"
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "amasum_hf"
if not DATA_DIR.exists():
    RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "amasum"
    save_dataset(DATA_DIR, RAW_DATA_PATH)

ds = load_from_disk(DATA_DIR)
# print(ds)
# print(ds["train"][0])

# create the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# create the model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

def tokenize_function(batch):
    sources = [" <doc-sep> ".join(doc) for doc in batch["documents"]]
    # 4096 is the max token length for PRIMERA
    model_inputs = tokenizer(sources, max_length=4096, truncation=True)

    labels = tokenizer(text_target=batch["verdict"], max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    # PRIMERA expects global attention on the first token and each <doc-sep
    doc_sep_id = tokenizer.convert_tokens_to_ids("<doc-sep>")
    model_inputs["global_attention_mask"] = [
        [1 if i == 0 or token_id == doc_sep_id else 0 for i, token_id in enumerate(ids)]
        for ids in model_inputs["input_ids"]
    ]

    return model_inputs

tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)
tokenized.set_format("torch")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


