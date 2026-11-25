# Training on Kaggle

How to fine-tune PRIMERA (`src/train.py`) on a Kaggle GPU notebook.

## 1. Notebook settings (right sidebar)

- **Accelerator:** GPU T4 x2.
- **Internet:** On. It's needed for the model download, pip, NLTK and ROUGE, and requires phone verification.
- **Inputs:** the `src` dataset and the `amasum_hf` dataset. Add the previous notebook output too if you're resuming (step 6).

## 2. Find your input paths

```python
!ls /kaggle/input
!ls /kaggle/input/*
```

Write down two folders:

- **`SRC_IN`**: the folder that contains `train.py` and `prepare_dataset.py`.
- **`DATA_IN`**: the folder that contains `dataset_dict.json`, which is `amasum_hf`.

Everything under `/kaggle/input` is read-only. You work in `/kaggle/working`, which is writable and saved as output when you commit.

## 3. Set up the working folder

```python
SRC_IN  = "/kaggle/input/<src-dataset>/src"          # adjust
DATA_IN = "/kaggle/input/<data-dataset>/amasum_hf"   # adjust

# code: copy so it's editable
!rm -rf /kaggle/working/src
!cp -r {SRC_IN} /kaggle/working/src

# data: symlink (recommended, no extra 2 GB in the output)
# copy to scratch space: writable, and not saved as output
!mkdir -p /kaggle/temp/datasets
!rm -rf /kaggle/temp/datasets/amasum_hf
!cp -r {DATA_IN} /kaggle/temp/datasets/amasum_hf
!ls /kaggle/temp/datasets/amasum_hf

import os
os.environ["DATA_DIR"] = "/kaggle/temp/datasets/amasum_hf"
```

Afterwards the layout looks like this:

```
/kaggle/working/
├── src/                  ← copy of the code (PROJECT_DIR = /kaggle/working)
├── datasets/amasum_hf →  symlink to the input
├── checkpoints/          ← created automatically during training
└── models/               ← final model, created at the end
```

With this layout, **no environment variables are needed**. Every default path in `train.py` resolves correctly.

There are two alternatives for the data:

- **Copy instead of linking:** `!cp -r {DATA_IN} /kaggle/working/datasets/`. This adds 2 GB to every saved output. If you do it, add `!rm -rf /kaggle/working/datasets` at the end of the notebook.
- **Skip the `datasets/` folder** and point the script at the input directly: `import os; os.environ["DATA_DIR"] = DATA_IN`.

## 4. Install missing packages

```python
!pip install -q evaluate rouge_score
```

## 5. (Optional) Edit the code

Edit the copy in `/kaggle/working/src`, never the input. For small tweaks:

```python
!sed -i 's/num_epochs = 3/num_epochs = 2/' /kaggle/working/src/train.py
!sed -i 's/batch_size=2, collate_fn/batch_size=1, collate_fn/' /kaggle/working/src/train.py   # if you run out of memory
!grep -n "num_epochs =\|batch_size" /kaggle/working/src/train.py
```

To replace a whole file, start a cell with `%%writefile /kaggle/working/src/train.py` and paste the full file. For lasting changes, edit locally and upload a **New Version** of the `src` dataset. These edits are lost each session, so keep the cells in the notebook.

## 6. (Optional) Resume from a previous run

Add the previous version's output via **Add Input → Your Work → this notebook**, then run:

```python
!mkdir -p /kaggle/working/checkpoints
!cp /kaggle/input/<previous-output>/checkpoints/*.pt /kaggle/working/checkpoints/
```

The script automatically picks up the newest `step_*.pt` and skips the batches it already trained on.

## 7. Train

On both GPUs:

```python
!cd /kaggle/working/src && accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train.py
```

Or on a single GPU:

```python
!cd /kaggle/working/src && python train.py
```

Optional overrides go in a cell before training:

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `<project>/datasets/amasum_hf` | Preprocessed HF dataset |
| `CKPT_DIR` | `<project>/checkpoints` | Training checkpoints |
| `FINAL_DIR` | `<project>/models` | Final model + tokenizer |
| `VAL_SUBSET` | `300` | Validation samples per epoch (`0` = full set) |

```python
import os
os.environ["VAL_SUBSET"] = "100"   # fewer validation samples, so faster epochs
```

## 8. Run it as a commit

Click **Save Version → Save & Run All (Commit)**. The run continues in the background for up to 12 hours, even if you close the browser. An interactive session stops when you go idle.

## 9. Outputs

After the commit finishes, its **Output** tab contains:

- `checkpoints/step_XXXXXXX.pt`: the last 2 checkpoints, saved every 500 steps and at the end of each epoch. They hold the trainable weights plus optimizer and scheduler state, for resuming.
- `models/`: the full fine-tuned model plus tokenizer, about 1.8 GB. Load it with:

  ```python
  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  path = "/kaggle/input/<this-notebook-output>/models"
  tokenizer = AutoTokenizer.from_pretrained(path)
  model = AutoModelForSeq2SeqLM.from_pretrained(path)
  ```

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for the dataset or `amasum` | The symlink or `DATA_DIR` is wrong. Check that `ls /kaggle/working/datasets/amasum_hf` shows `dataset_dict.json`. |
| CUDA out of memory | Set `batch_size=1` (step 5). |
| Model download or pip fails | Internet is off. |
| Epochs too slow | Lower `VAL_SUBSET`. |
| Run killed at 12 hours | Resume (step 6) and commit again. |
| Checkpoints missing after the run | The run was interactive, not committed (step 8). |
