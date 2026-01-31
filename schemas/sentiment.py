# schemas/sentiment.py
from pydantic import BaseModel
from typing import List

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    confidence: float

class BatchSentimentRequest(BaseModel):
    texts: List[str]
