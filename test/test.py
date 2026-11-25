"""
metrics: {'rouge1': np.float64(23.47), 'rouge2': np.float64(5.17), 'rougeL': np.float64(17.94), 'rougeLsum': np.float64(18.66), 'test_loss': 2.8172}
"""

from datasets import load_from_disk
from pathlib import Path
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm.auto import tqdm

from accelerate import Accelerator

import evaluate
import nltk

PROJECT_DIR = Path(__file__).resolve().parent.parent
# override with env vars on Kaggle, e.g. DATA_DIR=/kaggle/working/amasum_hf
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_DIR / "datasets" / "amasum_hf"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_DIR / "models"))

ds = load_from_disk(DATA_DIR)["test"]

# fp16 needs a GPU; fall back to full precision on CPU
accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")

# create the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# create the model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

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

with accelerator.main_process_first():
    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
tokenized.set_format("torch")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
def collate_function(features):
    global_masks = [f.pop("global_attention_mask") for f in features]
    batch = data_collator(features)
    batch["global_attention_mask"] = pad_sequence(global_masks, batch_first=True, padding_value=0)
    return batch

test_dataloader = DataLoader(tokenized, shuffle=False, batch_size=2, collate_fn=collate_function)
model, test_dataloader = accelerator.prepare(model, test_dataloader)

# evaluation: test loss + ROUGE on generated summaries
nltk.download("punkt_tab", quiet=True)
rouge = evaluate.load("rouge")

def postprocess(texts):
    # rougeLsum expects one sentence per line
    return ["\n".join(nltk.sent_tokenize(t.strip())) for t in texts]

def evaluate_model(dataloader, max_new_tokens=256, num_beams=4):
    model.eval()
    base = accelerator.unwrap_model(model)  # generate() lives on the unwrapped model
    losses = []

    for batch in tqdm(dataloader, desc="testing", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            # validation loss (teacher-forced)
            loss = model(**batch).loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].size(0))))

            # real summaries; PRIMERA needs the global attention mask here too
            # the unwrapped model isn't autocast-wrapped, so enable fp16 explicitly
            with accelerator.autocast():
                generated = base.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    global_attention_mask=batch["global_attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )

        # pad to the same length on every process before gathering
        generated = accelerator.pad_across_processes(generated, dim=1, pad_index=tokenizer.pad_token_id)
        labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=-100)
        generated, labels = accelerator.gather_for_metrics((generated, labels))

        # the collator puts -100 in the labels, which the tokenizer can't decode
        labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)

        preds = postprocess(tokenizer.batch_decode(generated, skip_special_tokens=True))
        refs = postprocess(tokenizer.batch_decode(labels, skip_special_tokens=True))
        rouge.add_batch(predictions=preds, references=refs)

    results = {k: round(v * 100, 2) for k, v in rouge.compute(use_stemmer=True).items()}
    results["test_loss"] = round(torch.cat(losses).mean().item(), 4)
    return results

metrics = evaluate_model(test_dataloader)
accelerator.print(f"metrics: {metrics}")

accelerator.wait_for_everyone()
accelerator.end_training()