from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Sentiment Analysis API")

df = pd.read_csv("IMDB Dataset.csv")

df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

X = df["review"]
y = df["sentiment"]

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_tfidf, y)

class ReviewData(BaseModel):
    review: str

@app.post("/predict/sentiment")
def predict_sentiment(data: ReviewData):
    review_tfidf = tfidf.transform([data.review])

    prediction = model.predict(review_tfidf)[0]
    probability = model.predict_proba(review_tfidf)[0]

    return {
        "Sentiment": "positive" if prediction == 1 else "negative",
        "Positive_Probability": round(float(probability[1]), 3),
        "Negative_Probability": round(float(probability[0]), 3)
    }