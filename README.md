# 🎬 YouTube Sentiment Insights

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-brightgreen?style=for-the-badge&logo=leaflet&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-black?style=for-the-badge&logo=flask&logoColor=white)
![Chrome](https://img.shields.io/badge/Chrome-Extension-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![DagsHub](https://img.shields.io/badge/DagsHub-MLflow%20Remote-orange?style=for-the-badge&logo=dagshub&logoColor=white)

**A production-ready MLOps pipeline that classifies YouTube comments as positive, negative, or neutral — surfaced directly inside the browser via a Chrome Extension.**

[🎥 Watch Demo](#-demo) · [🚀 Quick Start](#-getting-started) · [📊 Results](#-results) · [🌐 API Docs](#-api-reference)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [MLOps Architecture](#-mlops-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Chrome Extension](#-chrome-extension)
- [ML Pipeline](#-ml-pipeline-dvc)
- [Experiment Tracking](#-experiment-tracking-mlflow)
- [API Reference](#-api-reference)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [MLOps Skills Demonstrated](#-mlops-skills-demonstrated)

---

## 🔍 Overview

**YouTube Sentiment Insights** is a full end-to-end ML system that classifies YouTube comments into **positive**, **negative**, or **neutral** — and surfaces results directly in the browser via a **Chrome Extension**.

The project demonstrates a complete **MLOps workflow**: data versioning with DVC, experiment tracking with MLflow (hosted on **DagsHub**), model serving with Flask, containerization with Docker, and automated cloud deployment on AWS — all triggered via GitHub Actions CI/CD.

### Key Highlights

- ✅ **LightGBM** classifier with TF-IDF text preprocessing pipeline
- ✅ **Chrome Extension** — analyze any YouTube video's comments without leaving the browser
- ✅ **Flask REST API** for real-time inference
- ✅ **MLflow** experiment tracking hosted on **DagsHub** — full run history, metrics & artifacts logged remotely
- ✅ **Dockerized** for reproducible, portable deployment
- ✅ **AWS** cloud deployment (EC2 + ECR)
- ✅ **DVC** for data, model, and pipeline versioning
- ✅ **GitHub Actions** for fully automated CI/CD

---

## 🎥 Demo

[![Watch Demo](https://github.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis/raw/main/demo.png)](https://youtu.be/vyts7NzzUWk)

> Click the image to watch the full demo on YouTube ↑

---

## 🏗️ MLOps Architecture

```
                     ┌─────────────────────────────┐
                     │       Chrome Extension        │
                     │  (yt-chrome-plugin-frontend)  │
                     └─────────────┬────────────────┘
                                   │ HTTP Request
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline                             │
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐  │
│  │   Data   │───▶│  Train   │───▶│ Artifacts │───▶│  Flask API  │  │
│  │  (DVC)   │    │LightGBM  │    │  /model   │    │  :5000      │  │
│  └──────────┘    └────┬─────┘    └───────────┘    └──────┬──────┘  │
│        │              │                                   │         │
│  DVC Remote     ┌─────▼──────┐                    ┌──────▼──────┐  │
│  Storage        │   MLflow   │                    │   Docker    │  │
│                 │  EC2:5001  │◀── Experiment ──   │  Container  │  │
│                 │  S3 Bucket │    Tracking         └──────┬──────┘  │
│                 └────────────┘                            │         │
│                                         GitHub Actions    │         │
│                                         CI/CD ───────────▶│         │
│                                                    ┌──────▼──────┐  │
│                                                    │  AWS ECR +  │  │
│                                                    │  AWS EC2    │  │
│                                                    └─────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **ML Model** | LightGBM, Scikit-learn, TF-IDF |
| **NLP & Preprocessing** | NLTK, Regex, Custom Text Pipeline |
| **API Serving** | Flask, Gunicorn, Jinja2 Templates |
| **Frontend / Extension** | Chrome Extension (JS, HTML, CSS) |
| **Experiment Tracking** | MLflow (hosted on DagsHub) |
| **Containerization** | Docker, `.dockerignore` |
| **Data & Model Versioning** | DVC (`dvc.yaml`, `dvc.lock`, `.dvc/`) |
| **Experiment Config** | `params.yaml` |
| **CI/CD** | GitHub Actions (`.github/workflows/`) |
| **Cloud Deployment** | AWS EC2, AWS ECR, AWS S3 |
| **Version Control** | Git, GitHub |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis/
│
├── .github/
│   └── workflows/              # GitHub Actions CI/CD pipeline
│
├── .dvc/                       # DVC configuration & cache
├── dvc.yaml                    # DVC pipeline stage definitions
├── dvc.lock                    # Locked pipeline state (reproducibility)
├── params.yaml                 # Model hyperparameters & config
│
├── src/                        # Core ML source code
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   └── model_evaluation.py
│
├── Note-books/                 # EDA & experimentation notebooks
│
├── artifacts/                  # DVC-tracked model & data artifacts
│   ├── model/
│   └── data/
│
├── flask_api/                  # Flask REST API
│   └── app.py
│
├── templates/                  # Jinja2 HTML templates
│
├── yt-chrome-plugin-frontend/  # 🔌 Chrome Extension source
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── background.js
│
├── confusion_matrix_Test Data.png
├── demo.png
├── Dockerfile
├── .dockerignore
├── .dvcignore
├── setup.py
├── requirements.txt
└── README.md
```

---

## 🔌 Chrome Extension

One of the standout features of this project is the **Chrome Extension** — it brings the ML model directly into the browser, letting users analyze the sentiment of any YouTube video's comments without leaving the page.

### How It Works

1. User opens any YouTube video
2. Clicks the extension icon in Chrome
3. Extension fetches the video's comments
4. Sends them to the deployed Flask API on AWS EC2
5. Displays a sentiment breakdown **(positive / negative / neutral)** directly in the popup

### Install the Extension (Developer Mode)

```bash
1. Open Chrome → go to chrome://extensions/
2. Enable "Developer mode" (top-right toggle)
3. Click "Load unpacked"
4. Select the yt-chrome-plugin-frontend/ folder
5. Open any YouTube video and click the extension icon 🎉
```

---

## 🔄 ML Pipeline (DVC)

The pipeline is fully defined in `dvc.yaml` for end-to-end reproducibility. Each stage is tracked and cached — only changed stages are re-run.

```yaml
# Pipeline stages defined in dvc.yaml
stages:
  data_ingestion:       # Fetch & store raw YouTube comment data
  data_transformation:  # Clean text, extract TF-IDF features
  model_trainer:        # Train & tune LightGBM classifier
  model_evaluation:     # Compute metrics, save confusion matrix
```

**Reproduce the full pipeline:**
```bash
dvc repro
```

**Run experiments with different hyperparameters:**
```bash
dvc exp run --set-param model.num_leaves=63
dvc exp show
```

**Push/pull data & model artifacts:**
```bash
dvc push   # Upload artifacts to remote storage
dvc pull   # Download tracked artifacts
```

---

## 📈 Experiment Tracking (MLflow on DagsHub)

Experiment tracking is handled by **MLflow**, hosted remotely on **DagsHub** — giving full visibility into every training run with zero infrastructure setup.

🔗 **[View Live MLflow Experiments on DagsHub](https://dagshub.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis.mlflow/#/experiments)**

```
MLflow Tracking Server (DagsHub)
  ├── Host: DagsHub Remote MLflow Server
  ├── Artifact Store: DagsHub artifact storage
  └── Tracks: parameters, metrics, model artifacts per run
```

**Logged per experiment run:**
- Hyperparameters from `params.yaml` (e.g., `num_leaves`, `learning_rate`, `n_estimators`)
- Evaluation metrics: Accuracy, F1-Score, Precision, Recall
- Model artifacts and confusion matrix

**Configure MLflow to log to DagsHub:**
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis.mlflow
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

Or in Python:
```python
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis.mlflow")
```

---

## 🌐 API Reference

**Base URL:** `http://<your-ec2-ip>:5000`

### `POST /predict`

Predict sentiment for a YouTube comment.

**Request:**
```json
{
  "comment": "This video is absolutely amazing, I learned so much!"
}
```

**Response:**
```json
{
  "comment": "This video is absolutely amazing, I learned so much!",
  "sentiment": "positive",
  "confidence": 0.94
}
```

### `GET /health`

```json
{ "status": "healthy", "model": "loaded" }
```

---

## ⚙️ CI/CD Pipeline

Every push to `main` automatically triggers the full deployment pipeline via **GitHub Actions**:

```
Push to main branch
        │
        ▼
┌───────────────────┐
│   1. Run Tests    │  ← Unit & integration tests
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  2. Build Docker  │  ← docker build -t sentiment-api .
│      Image        │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  3. Push to ECR   │  ← AWS Elastic Container Registry
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  4. Deploy to EC2 │  ← SSH → docker pull → docker run
└───────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker
- DVC: `pip install dvc`
- AWS CLI (for cloud deployment)
- Chrome Browser (for the extension)

### 1. Clone the repo

```bash
git clone https://github.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis.git
cd End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull data & model artifacts

```bash
dvc pull
```

### 4. Reproduce the ML pipeline

```bash
dvc repro
```

### 5. Run locally with Docker

```bash
docker build -t sentiment-api .
docker run -p 5000:5000 sentiment-api
```

### 6. Test the API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "This is the best tutorial I have ever watched!"}'
```

---

## 📊 Results

| Metric | Score |
|---|---|
| **Accuracy** | ~XX% |
| **F1-Score (Macro)** | ~XX% |
| **Precision** | ~XX% |
| **Recall** | ~XX% |

> Metrics tracked and logged via **MLflow on DagsHub** — [view full experiment history →](https://dagshub.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis.mlflow/#/experiments)

[![Confusion Matrix](https://github.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis/raw/main/confusion_matrix_Test%20Data.png)](https://github.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis/blob/main/confusion_matrix_Test%20Data.png)

---

## 🧠 MLOps Skills Demonstrated

| Skill | How It's Applied |
|---|---|
| **Data Versioning** | DVC tracks raw data, features & model artifacts in `artifacts/` |
| **Pipeline Reproducibility** | `dvc repro` re-runs only changed stages defined in `dvc.yaml` |
| **Experiment Tracking** | MLflow hosted on **DagsHub** — logs params, metrics & artifacts per run · [View Experiments ↗](https://dagshub.com/omarhatem44/End-to-End-MLOps-System-for-Real-Time-YouTube-Sentiment-Analysis.mlflow/#/experiments) |
| **Artifact Storage** | MLflow artifacts stored via DagsHub remote storage |
| **Model Serving** | Production Flask API with health check endpoint |
| **Containerization** | Dockerfile + `.dockerignore` for consistent dev/prod environment |
| **CI/CD Automation** | GitHub Actions builds, tests, and deploys on every push to `main` |
| **Cloud Deployment** | Docker image pushed to AWS ECR, served live on AWS EC2 |
| **Product Thinking** | Chrome Extension bridges the ML model to a real user-facing product |

---

## 👤 Author

**Omar Hatem**

- 🎓 Computer Science Student — Modern Academy for Computer Science, Cairo
- 💼 ML Engineer | MLOps Enthusiast
- 🔗 [GitHub](https://github.com/omarhatem44) · [LinkedIn](https://linkedin.com/in/your-profile)

---

<div align="center">

*Built end-to-end with production MLOps practices in mind 🚀*

</div>
