import mlflow
import mlflow.sklearn
import dagshub

import pandas as pd
import optuna
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# 🔥 ربط DagsHub
dagshub.init(
    repo_owner="omarhatem44",
    repo_name="End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis",  # غيرها لو عايز repo تاني
    mlflow=True
)

# ✅ اسم experiment الجديد
mlflow.set_experiment("Experiment_6_lightgbm_detailed_hpt")

# Load data
df = pd.read_csv(
    r'D:\omar\MLOps\Youtube-Sentiment-Insights-Done\Note-books\dataset.csv'
).dropna(subset=['clean_comment'])

# Remap labels
df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
df = df.dropna(subset=['category'])

# Vectorization
ngram_range = (1, 3)
max_features = 1000

vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,
    max_features=max_features
)

X = vectorizer.fit_transform(df['clean_comment'])
y = df['category']

# 🔥 Handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

# 🔥 Logging function
def log_mlflow(model_name, model, params, trial_number):
    with mlflow.start_run():

        mlflow.set_tag("mlflow.runName", f"Trial_{trial_number}_LightGBM")
        mlflow.set_tag("platform", "DagsHub")
        mlflow.set_tag("experiment_type", "hpt_detailed")

        mlflow.log_param("algo_name", model_name)

        # Log params
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Train
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

        # Log model
        mlflow.sklearn.log_model(model, "lightgbm_model")

        return accuracy


# 🔥 Optuna objective
def objective_lightgbm(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)
    }

    model = LGBMClassifier(**params, random_state=42)

    return log_mlflow("LightGBM", model, params, trial.number)


# 🔥 Run Optuna
def run_optuna_experiment():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_lightgbm, n_trials=100)

    best_params = study.best_params

    # 🔥 Log best model separately
    with mlflow.start_run():

        mlflow.set_tag("mlflow.runName", "LightGBM_BEST_MODEL")
        mlflow.set_tag("platform", "DagsHub")

        for key, value in best_params.items():
            mlflow.log_param(key, value)

        best_model = LGBMClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # 🔥 Save vectorizer (مهم جداً)
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        mlflow.log_artifact("vectorizer.pkl")

        mlflow.sklearn.log_model(best_model, "best_lightgbm_model")


# Run
run_optuna_experiment()