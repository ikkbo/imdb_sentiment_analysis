# app.py（现在）
from fastapi import FastAPI
import torch
from transformers import BertTokenizer
from model.sentiment_model import SentimentAnalysisModel
from schemas.sentiment import SentimentRequest, SentimentResponse
from services.inference import predict_sentiment
from schemas.sentiment import BatchSentimentRequest
from services.inference import batch_predict
from fastapi.concurrency import run_in_threadpool


app = FastAPI()


@app.on_event("startup")
def startup_event():
    global tokenizer, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = SentimentAnalysisModel("bert-base-uncased")
    model.load_state_dict(torch.load("bert_imdb_sentiment.pth", map_location=device))
    model.to(device)
    model.eval()


@app.post("/predict", response_model=SentimentResponse)
async def predict_api(req: SentimentRequest):
    label, conf = await run_in_threadpool(
        predict_sentiment, req.text, tokenizer, model, device
    )
    return SentimentResponse(label=label, confidence=conf)



@app.post("/predict_batch")
async def predict_batch_api(req: BatchSentimentRequest):
    results = await run_in_threadpool(
        batch_predict, req.texts, tokenizer, model, device
    )
    return results



@app.get("/health")
def health():
    return {"status": "ok"}
