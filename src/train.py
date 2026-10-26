from datasets import load_from_disk
from download_dataset import save_dataset
from pathlib import Path

path = Path("../datasets/google-reviews-dataset")
if not path.exists():
    save_dataset()


