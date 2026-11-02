from datasets import load_from_disk
from prepare_dataset import save_dataset
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, get_scheduler
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm.auto import tqdm

from accelerate import Accelerator

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
def collate_function(features):
    global_masks = [f.pop("global_attention_mask") for f in features]
    batch = data_collator(features)
    batch["global_attention_mask"] = pad_sequence(global_masks, batch_first=True, padding_value=0)
    return batch

train_dataloader = DataLoader(tokenized["train"], shuffle=True, batch_size=2, collate_fn=collate_function)
val_dataloader = DataLoader(tokenized["valid"], shuffle=False, batch_size=2, collate_fn=collate_function)
test_dataloader = DataLoader(tokenized["test"], shuffle=False, batch_size=2, collate_fn=collate_function)

# only allow training for the summarizing head
for param in model.parameters():
    param.requires_grad = False
for params in model.led.decoder.layers[-2:].parameters():
    params.requires_grad = True
