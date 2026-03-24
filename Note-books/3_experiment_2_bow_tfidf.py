import mlflow
import mlflow.sklearn
import dagshub

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle

# 🔥 ربط DagsHub
dagshub.init(
    repo_owner="omarhatem44",
    repo_name="End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis",  # غير لو عندك repo تاني
    mlflow=True
)

# تحديد experiment
mlflow.set_experiment("Experiment_2_bow_tfidf")

# Load data
df = pd.read_csv(
    r'D:\omar\MLOps\Youtube-Sentiment-Insights-Done\Note-books\dataset.csv'
).dropna(subset=['clean_comment'])


# Function to run experiment
def run_experiment(vectorizer_type, ngram_range, vectorizer_max_features, vectorizer_name):

    if vectorizer_type == "BoW":
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=vectorizer_max_features
        )
    else:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=vectorizer_max_features
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_comment'],
        df['category'],
        test_size=0.2,
        random_state=42,
        stratify=df['category']
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    with mlflow.start_run():

        mlflow.set_tag("mlflow.runName", f"{vectorizer_name}_{ngram_range}_RF")
        mlflow.set_tag("platform", "DagsHub")
        mlflow.set_tag("experiment_type", "feature_engineering")

        # Parameters
        n_estimators = 200
        max_depth = 15

        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", vectorizer_max_features)
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

        # Confusion Matrix (اسم مختلف لكل run)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"CM: {vectorizer_name}, {ngram_range}")

        cm_filename = f"cm_{vectorizer_name}_{ngram_range}.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()

        # 🔥 Save + Log Vectorizer (مهم جداً للـ deployment)
        vec_filename = f"vectorizer_{vectorizer_name}_{ngram_range}.pkl"
        with open(vec_filename, "wb") as f:
            pickle.dump(vectorizer, f)

        mlflow.log_artifact(vec_filename)

        # Log model
        mlflow.sklearn.log_model(
            model,
            f"rf_model_{vectorizer_name}_{ngram_range}"
        )


# Run experiments
ngram_ranges = [(1, 1), (1, 2), (1, 3)]
max_features = 5000

for ngram_range in ngram_ranges:
    run_experiment("BoW", ngram_range, max_features, "BoW")
    run_experiment("TF-IDF", ngram_range, max_features, "TF-IDF")