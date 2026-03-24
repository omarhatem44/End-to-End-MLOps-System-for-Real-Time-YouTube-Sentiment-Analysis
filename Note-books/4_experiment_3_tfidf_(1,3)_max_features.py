import mlflow
import mlflow.sklearn
import dagshub

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

# 🔥 ربط DagsHub بدل MLflow server
dagshub.init(
    repo_owner="omarhatem44",
    repo_name="End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis",
    mlflow=True
)

# ✅ اسم experiment الجديد
mlflow.set_experiment("Experiment_3_tfidf_(1,3)_max_features")

# Load data
df = pd.read_csv(
    r'D:\omar\MLOps\Youtube-Sentiment-Insights-Done\Note-books\dataset.csv'
).dropna(subset=['clean_comment'])


def run_experiment_tfidf_max_features(max_features):
    ngram_range = (1, 3)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
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

        mlflow.set_tag("mlflow.runName", f"TFIDF_(1,3)_max_{max_features}")
        mlflow.set_tag("platform", "DagsHub")

        # Parameters
        n_estimators = 200
        max_depth = 15

        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", max_features)
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

        # 🔥 Confusion Matrix (unique name)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d")

        cm_filename = f"cm_tfidf_(1,3)_{max_features}.png"
        plt.title(f"CM TF-IDF (1,3) max={max_features}")
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()

        # 🔥 Save vectorizer
        vec_filename = f"tfidf_vec_(1,3)_{max_features}.pkl"
        with open(vec_filename, "wb") as f:
            pickle.dump(vectorizer, f)

        mlflow.log_artifact(vec_filename)

        # Log model
        mlflow.sklearn.log_model(
            model,
            f"rf_tfidf_(1,3)_{max_features}"
        )


# Run experiments
max_features_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

for max_features in max_features_values:
    run_experiment_tfidf_max_features(max_features)