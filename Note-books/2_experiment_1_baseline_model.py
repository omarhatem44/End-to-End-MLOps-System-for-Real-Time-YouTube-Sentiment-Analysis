import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# 🔥 Initialize DagsHub (بديل MLflow server)
dagshub.init(
    repo_owner="omarhatem44",
    repo_name="End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis",  # غيرها لو عايز repo تاني
    mlflow=True
)

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[~(df['clean_comment'].str.strip() == '')]

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_comment(comment):
    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment

df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)

# Vectorization
vectorizer = CountVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_comment']).toarray()
y = df['category']

# ❌ شيلنا mlflow.set_tracking_uri
# ✅ بس بنحدد experiment عادي
mlflow.set_experiment("RF Baseline DagsHub")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():

    mlflow.set_tag("mlflow.runName", "RF_DagsHub_Run")
    mlflow.set_tag("platform", "DagsHub")

    # Parameters
    n_estimators = 200
    max_depth = 15

    mlflow.log_param("vectorizer", "CountVectorizer")
    mlflow.log_param("max_features", vectorizer.max_features)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    report = classification_report(y_test, y_pred, output_dict=True)

    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric}", value)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Log dataset
    df.to_csv("dataset.csv", index=False)
    mlflow.log_artifact("dataset.csv")

print(f"Accuracy: {accuracy}")