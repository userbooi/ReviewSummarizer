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
from pprint import pprint

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
# to include padding for the global attention mask
def collate_function(features):
    global_masks = [f.pop("global_attention_mask") for f in features]
    batch = data_collator(features)
    batch["global_attention_mask"] = pad_sequence(global_masks, batch_first=True, padding_value=0)
    return batch

train_dataloader = DataLoader(tokenized["train"], shuffle=True, batch_size=2, collate_fn=collate_function)
val_dataloader = DataLoader(tokenized["valid"], shuffle=False, batch_size=2, collate_fn=collate_function)
test_dataloader = DataLoader(tokenized["test"], shuffle=False, batch_size=2, collate_fn=collate_function)

# only allow training for the last two layers head
for param in model.parameters():
    param.requires_grad = False
for params in model.led.decoder.layers[-2:].parameters():
    params.requires_grad = True

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=5e-5)

# custom mode to stops dropout from frozen layers
def set_train_mode(model):
    model.train()
    # led isn't in an accelerator wrapped model
    base = accelerator.unwrap_model(model)
    base.led.encoder.eval()
    base.led.decoder.layers[:-2].eval()

accelerator = Accelerator()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

val_dataloader, test_dataloader = accelerator.prepare(
    val_dataloader, test_dataloader
)

# calculate training steps
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.05 * num_training_steps),
    num_training_steps=num_training_steps
)

# checkpointing: only the trainable params + optimizer/scheduler state
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
KEEP_LAST = 2  # how many checkpoints to keep on disk

def save_checkpoint(step, epoch):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    base = accelerator.unwrap_model(model)
    trainable_state = {
        name: p.detach().cpu() for name, p in base.named_parameters() if p.requires_grad
    }
    ckpt = {
        "model": trainable_state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    accelerator.save(ckpt, CKPT_DIR / f"step_{step:07d}.pt")

    # delete older checkpoints (zero-padded names sort chronologically)
    for old in sorted(CKPT_DIR.glob("step_*.pt"))[:-KEEP_LAST]:
        old.unlink()

def load_checkpoint(path):
    """Load trainable weights + optimizer/scheduler state on top of the base model. Returns the step."""
    ckpt = torch.load(path, map_location="cpu")
    base = accelerator.unwrap_model(model)
    # strict=False: the frozen weights aren't in the file, they come from the base checkpoint
    _, unexpected = base.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    return ckpt["step"]

def latest_checkpoint():
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"))
    return ckpts[-1] if ckpts else None

save_every = 500  # steps
resume_from = None  # e.g. latest_checkpoint()

global_step = 0
if resume_from is not None:
    global_step = load_checkpoint(resume_from)
    accelerator.print(f"Resumed from {resume_from} at step {global_step}")

# get the cool progress bar yay
progress_bar = tqdm(range(num_training_steps), initial=global_step)

# the dimensions of output.logits is (batch-size, target_seq_length, vocab_size)
# batch = next(iter(train_dataloader))
# output = model(**batch)
# predictions = torch.argmax(output.logits, dim=1)
# pprint(output.logits)
# print(output.logits.shape)

# training loop
steps_per_epoch = len(train_dataloader)
for epoch in range(global_step // steps_per_epoch, num_epochs):
    set_train_mode(model)

    # when resuming mid-epoch, skip the batches already trained on
    skip = global_step - epoch * steps_per_epoch
    epoch_dataloader = accelerator.skip_first_batches(train_dataloader, skip) if skip > 0 else train_dataloader

    for batch in epoch_dataloader:
        loss = model(**batch).loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()

        progress_bar.update(1)
        global_step += 1

        if global_step % save_every == 0:
            save_checkpoint(global_step, epoch)

    # always checkpoint at the end of an epoch
    save_checkpoint(global_step, epoch)

    # validate after each epoch
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)