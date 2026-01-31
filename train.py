import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data.imdb_data import  IMDBDataset
from model.sentiment_model import SentimentAnalysisModel


def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)

            preds.extend(predictions.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    acc = accuracy_score(labels, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    return acc


def train():
    # ================== 超参数 ==================
    model_name = "bert-base-uncased"
    batch_size = 8
    max_length = 256
    lr = 2e-5
    epochs = 3
    # ===========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = IMDBDataset("train", tokenizer, max_length)
    test_dataset = IMDBDataset("test", tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SentimentAnalysisModel(model_name).to(device)
    print("Model device:", next(model.parameters()).device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ================== 训练 ==================
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} Training Loss: {avg_loss:.4f}")

        evaluate(model, test_loader, device)

    # ================== 保存模型 ==================
    torch.save(model.state_dict(), "bert_imdb_sentiment.pth")
    print("Model saved.")


if __name__ == "__main__":
    train()
