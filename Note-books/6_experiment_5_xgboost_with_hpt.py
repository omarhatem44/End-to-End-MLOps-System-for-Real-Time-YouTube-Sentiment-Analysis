import mlflow
import mlflow.sklearn
import dagshub

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import pandas as pd
import pickle

# 🔥 DagsHub integration
dagshub.init(
    repo_owner="omarhatem44",
    repo_name="End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis",  # غيرها لو عايز repo تاني
    mlflow=True
)

# ✅ Experiment name
mlflow.set_experiment("Experiment_5_xgboost_with_hpt")

# Load data
df = pd.read_csv(
    r'D:\omar\MLOps\Youtube-Sentiment-Insights-Done\Note-books\dataset.csv'
).dropna(subset=['clean_comment'])

# Remap labels
df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
df = df.dropna(subset=['category'])

# Config
ngram_range = (1, 3)
max_features = 10000

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_comment'],
    df['category'],
    test_size=0.2,
    random_state=42,
    stratify=df['category']
)

# Vectorization
vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,
    max_features=max_features
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔥 Handle imbalance
smote = SMOTE(random_state=42)
X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)


# 🔥 Optuna objective
def objective_xgboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )

    preds = model.fit(X_train_vec, y_train).predict(X_test_vec)
    return accuracy_score(y_test, preds)


# 🔥 Run Optuna + log best model
def run_optuna_experiment():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_xgboost, n_trials=30)

    best_params = study.best_params

    with mlflow.start_run():

        mlflow.set_tag("mlflow.runName", "XGBoost_Optuna_Best")
        mlflow.set_tag("platform", "DagsHub")
        mlflow.set_tag("experiment_type", "hpt")

        # 🔥 Log best params
        mlflow.log_param("n_estimators", best_params['n_estimators'])
        mlflow.log_param("learning_rate", best_params['learning_rate'])
        mlflow.log_param("max_depth", best_params['max_depth'])

        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("imbalance", "SMOTE")

        # Train best model
        best_model = XGBClassifier(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            random_state=42
        )

        best_model.fit(X_train_vec, y_train)
        y_pred = best_model.predict(X_test_vec)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        report = classification_report(y_test, y_pred, output_dict=True)

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # 🔥 Save vectorizer
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        mlflow.log_artifact("vectorizer.pkl")

        # Log model
        mlflow.sklearn.log_model(best_model, "xgboost_best_model")


# Run
run_optuna_experiment()