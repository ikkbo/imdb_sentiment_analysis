import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class IMDBDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=256):
        print(f"Loading IMDB {split} dataset...")
        self.dataset = load_dataset("imdb")[split]
        print(f"IMDB {split} loaded.")
        self.encodings = tokenizer(
            self.dataset["text"],
            truncation=True,
            padding=True,
            max_length=max_length
        )
        self.labels = self.dataset["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
