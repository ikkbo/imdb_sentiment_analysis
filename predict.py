import torch
from transformers import BertTokenizer
from model.sentiment_model import SentimentAnalysisModel


# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. 加载模型结构
model = SentimentAnalysisModel("bert-base-uncased")

# 3. 加载训练好的权重
model.load_state_dict(
    torch.load("bert_imdb_sentiment.pth", map_location=device)
)

model.to(device)
model.eval()  # ⚠️ 非常重要

print("Model loaded successfully.")

def predict_sentiment(text):
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

    label_map = {0: "Negative 😡", 1: "Positive 😊"}
    return label_map[pred], probs[0][pred].item()

def batch_predict(texts):
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
        preds = torch.argmax(outputs, dim=1)

    return preds.cpu().tolist()


if __name__ == "__main__":
    # text = "This movie was absolutely amazing, I loved it!"
    texts = [
        "This movie is terrible.",
        "I really enjoyed this film!",
        "Not bad, but could be better."
    ]

    results = batch_predict(texts)
    print(results)  # [0, 1, 1]
    # | 数值 | 含义 |
    # | -- | -------- |
    # | 0 | Negative |
    # | 1 | Positive |

    # label, confidence = predict_sentiment(texts)
    # print(f"Text: {text}")
    # print(f"Prediction: {label}, confidence: {confidence:.4f}")
