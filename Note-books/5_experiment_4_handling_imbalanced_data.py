import mlflow
import mlflow.sklearn
import dagshub

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

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
    repo_name="End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis",  # غيرها لو عايز repo تاني
    mlflow=True
)

# ✅ اسم experiment الجديد
mlflow.set_experiment("Experiment_4_handling_imbalanced_data")

# Load data
df = pd.read_csv(
    r'D:\omar\MLOps\Youtube-Sentiment-Insights-Done\Note-books\dataset.csv'
).dropna(subset=['clean_comment'])


def run_imbalanced_experiment(imbalance_method):
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

    # Handle imbalance
    if imbalance_method == 'class_weights':
        class_weight = 'balanced'
    else:
        class_weight = None

        if imbalance_method == 'oversampling':
            smote = SMOTE(random_state=42)
            X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

        elif imbalance_method == 'adasyn':
            adasyn = ADASYN(random_state=42)
            X_train_vec, y_train = adasyn.fit_resample(X_train_vec, y_train)

        elif imbalance_method == 'undersampling':
            rus = RandomUnderSampler(random_state=42)
            X_train_vec, y_train = rus.fit_resample(X_train_vec, y_train)

        elif imbalance_method == 'smote_enn':
            smote_enn = SMOTEENN(random_state=42)
            X_train_vec, y_train = smote_enn.fit_resample(X_train_vec, y_train)

    with mlflow.start_run():

        mlflow.set_tag("mlflow.runName", f"Imbalance_{imbalance_method}")
        mlflow.set_tag("platform", "DagsHub")
        mlflow.set_tag("experiment_type", "imbalance_handling")

        # Parameters
        n_estimators = 200
        max_depth = 15

        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("imbalance_method", imbalance_method)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight=class_weight
        )

        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        report = classification_report(y_test, y_pred, output_dict=True)

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Confusion Matrix (unique name)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d")

        cm_filename = f"cm_imbalance_{imbalance_method}.png"
        plt.title(f"CM - {imbalance_method}")
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()

        # 🔥 Save vectorizer (مهم جداً)
        vec_filename = f"tfidf_vec_imbalance_{imbalance_method}.pkl"
        with open(vec_filename, "wb") as f:
            pickle.dump(vectorizer, f)

        mlflow.log_artifact(vec_filename)

        # Log model
        mlflow.sklearn.log_model(
            model,
            f"rf_tfidf_imbalance_{imbalance_method}"
        )


# Run experiments
imbalance_methods = [
    'class_weights',
    'oversampling',
    'adasyn',
    'undersampling',
    'smote_enn'
]

for method in imbalance_methods:
    run_imbalanced_experiment(method)