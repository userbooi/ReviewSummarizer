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

## 10. Test the fine-tuned model

`test/test.py` scores the fine-tuned model on the AMASUM **test** split. It reports ROUGE-1/2/L/Lsum on generated summaries (beam search, 4 beams, up to 256 new tokens) and the teacher-forced `test_loss`. Use a fresh notebook, or the training notebook after training has finished.

### Inputs

Use the same settings as step 1, plus:

- the **training notebook's output**, via **Add Input → Your Work → <training notebook>**, which supplies `models/`;
- the **`test` folder**. The `src` dataset doesn't include it, so either upload `test/` as its own dataset or add it to a new version of the `src` dataset.

### Setup

```python
TEST_IN  = "/kaggle/input/<test-dataset>/test"                  # adjust
MODEL_IN = "/kaggle/input/<training-notebook-output>/models"     # adjust
DATA_IN  = "/kaggle/input/<data-dataset>/amasum_hf"              # adjust

!rm -rf /kaggle/working/test
!cp -r {TEST_IN} /kaggle/working/test
!pip install -q evaluate rouge_score

import os
os.environ["MODEL_DIR"] = MODEL_IN   # read-only is fine, the model is only loaded
os.environ["DATA_DIR"]  = DATA_IN
```

`test.py` uses the folder above `test/` as its project folder, so with no environment variables it looks for `/kaggle/working/models` and `/kaggle/working/datasets/amasum_hf`. If you test in the same notebook right after training, the model is already in `/kaggle/working/models`, so you only need `DATA_DIR`, as set in step 3.

### Run

```python
# both GPUs
!cd /kaggle/working/test && accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 test.py

# or a single GPU
!cd /kaggle/working/test && python test.py
```

The last line of the log is the result:

```
metrics: {'rouge1': ..., 'rouge2': ..., 'rougeL': ..., 'rougeLsum': ..., 'test_loss': ...}
```

Scores are percentages and `test_loss` is the mean cross-entropy loss. Copy the line into the notes at the top of `test/model_comparison.py` to keep a record.

### Runtime

`test.py` has no subset option, so it generates summaries for all **3,166** test examples. That can take hours on a T4, so run it as a commit (step 8). For a quick check first, limit it to the first N examples:

```python
!sed -i 's/\["test"\]/["test"].select(range(200))/' /kaggle/working/test/test.py
!grep -n 'load_from_disk' /kaggle/working/test/test.py
```

Use the full set for any number you report.

### Baseline comparison

`test/model_comparison.py` gives the reference point. It runs the **untuned** `allenai/PRIMERA-multinews` from the Hub on the **validation** split, limited to the first `VAL_SUBSET` (default 300) examples, which is the same subset training validates on. It needs Internet but no `MODEL_DIR`:

```python
!cd /kaggle/working/test && accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 model_comparison.py
```

To compare it with the fine-tuned model, use the per-epoch validation metrics from the training log, not the `test.py` numbers, since those come from a different split. The results recorded so far are at the top of `model_comparison.py`: a ROUGE-1 of 9.25 for the baseline against 23.91 after fine-tuning.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for the dataset or `amasum` | The symlink or `DATA_DIR` is wrong. Check that `ls /kaggle/working/datasets/amasum_hf` shows `dataset_dict.json`. |
| CUDA out of memory | Set `batch_size=1` (step 5). |
| Model download or pip fails | Internet is off. |
| Epochs too slow | Lower `VAL_SUBSET`. |
| Run killed at 12 hours | Resume (step 6) and commit again. |
| Checkpoints missing after the run | The run was interactive, not committed (step 8). |
| `test.py`: `OSError` / can't find `config.json` | `MODEL_DIR` is wrong. It must point at the folder that contains `config.json` and `model.safetensors`. |
| `test.py` takes too long | Test a subset first (step 10, Runtime), then run the full set as a commit. |
