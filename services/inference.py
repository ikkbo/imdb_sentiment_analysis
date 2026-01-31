# services/inference.py
import torch

def predict_sentiment(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label_map = {0: "Negative", 1: "Positive"}
    return label_map[pred], probs[0][pred].item()

def batch_predict(texts, tokenizer, model, device):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    label_map = {0: "Negative", 1: "Positive"}

    return [
        {
            "text": text,
            "label": label_map[p.item()],
            "confidence": probs[i][p].item()
        }
        for i, (text, p) in enumerate(zip(texts, preds))
    ]
