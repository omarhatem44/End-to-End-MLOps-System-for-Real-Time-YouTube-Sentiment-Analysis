# =========================================================
# Imports
# =========================================================

import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import io
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# Flask App Initialization
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# Text Preprocessing
# =========================================================

def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - {"not","but","however","no","yet"}

        comment = " ".join([w for w in comment.split() if w not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(w) for w in comment.split()])

        return comment

    except Exception as e:
        print("Preprocessing error:", e)
        return comment


# =========================================================
# Load Model
# =========================================================

def load_model(model_path, vectorizer_path):

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model(
    "artifacts/lgbm_model.pkl",
    "artifacts/tfidf_vectorizer.pkl"
)


# =========================================================
# Helper
# =========================================================

def extract_comments(data):

    comments = data.get("comments")

    if not comments:
        return None

    if isinstance(comments[0], dict):
        comments = [c.get("text","") for c in comments]

    return comments


# =========================================================
# Routes
# =========================================================

@app.route("/")
def home():
    return "YouTube Comment Sentiment API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    comments = extract_comments(data)

    if not comments:
        return jsonify({"error":"No comments provided"}),400

    try:

        preprocessed = [preprocess_comment(c) for c in comments]

        transformed = vectorizer.transform(preprocessed)

        dense = transformed.toarray()

        predictions = model.predict(dense).tolist()

        response = [
            {"comment":c,"sentiment":int(p)}
            for c,p in zip(comments,predictions)
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error":str(e)}),500


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.json
    comments = extract_comments(data)

    if not comments:
        return jsonify({"error":"No comments provided"}),400

    try:

        preprocessed = [preprocess_comment(c) for c in comments]

        transformed = vectorizer.transform(preprocessed)

        dense = transformed.toarray()

        predictions = model.predict(dense).tolist()

        counts = Counter(predictions)

        positive = counts.get(1,0)
        neutral = counts.get(0,0)
        negative = counts.get(-1,0)

        total = len(comments)

        result = {

            "positive":positive,
            "neutral":neutral,
            "negative":negative,
            "total":total,

            "percent_positive":round(positive/total*100,1),
            "percent_neutral":round(neutral/total*100,1),
            "percent_negative":round(negative/total*100,1),

            "predictions":predictions
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error":str(e)}),500


@app.route("/generate_chart", methods=["POST"])
def generate_chart():

    data = request.json
    counts = data.get("sentiment_counts")

    labels = ["Positive","Neutral","Negative"]

    sizes = [
        int(counts.get("1",0)),
        int(counts.get("0",0)),
        int(counts.get("-1",0))
    ]

    colors = ["#36A2EB","#C9CBCF","#FF6384"]

    plt.figure(figsize=(6,6))
    plt.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%")

    img = io.BytesIO()

    plt.savefig(img,format="png")
    img.seek(0)
    plt.close()

    return send_file(img,mimetype="image/png")


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():

    data = request.json
    comments = extract_comments(data)

    text = " ".join([preprocess_comment(c) for c in comments])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black"
    ).generate(text)

    img = io.BytesIO()

    wordcloud.to_image().save(img,format="PNG")

    img.seek(0)

    return send_file(img,mimetype="image/png")


# =========================================================
# Run Server
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
