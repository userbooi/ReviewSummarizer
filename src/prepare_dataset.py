from datasets import load_dataset, DatasetDict, Dataset
from pathlib import Path
import json
from pprint import pprint

# load the data from a specific split into a HF Dataset
def load_split(split_dir: Path) -> Dataset:
    rows = []

    for path in Path(split_dir).glob("*.json"):
        with open(path) as f:
            prod = json.load(f)

        reviews = [r["text"] for r in prod["customer_reviews"]]
        summ = prod["website_summaries"][0]
        rows.append({
            "documents": reviews,
            "ratings": [r.get("rating") for r in prod["customer_reviews"]],
            "verdict": summ["verdict"],
            "pros": " ".join(summ["pros"]),
            "cons": " ".join(summ["cons"]),
        })
    return Dataset.from_list(rows)

def save_dataset(DATA_SAVE_PATH: Path, RAW_DATA_PATH: Path) -> None:
    ds = DatasetDict({
        split: load_split(RAW_DATA_PATH / f"{split}")
        for split in ["train", "valid", "test"]
    })

    # check the data
    # print(ds)
    # pprint(ds["train"][:3])

    ds.save_to_disk(DATA_SAVE_PATH)
