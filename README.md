
# 🎬 YouTube Sentiment Insights

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-Model-brightgreen?style=for-the-badge&logo=leaflet&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-API-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS-Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white"/>
  <img src="https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge&logo=dvc&logoColor=white"/>
</p>

<p align="center">
  A <strong>production-ready MLOps pipeline</strong> for analyzing sentiment in YouTube comments — from data versioning and model training to containerized deployment on AWS with full CI/CD automation.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [MLOps Architecture](#-mlops-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [API Reference](#-api-reference)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Getting Started](#-getting-started)
- [Results](#-results)

---

## 🔍 Overview

**YouTube Sentiment Insights** is an end-to-end machine learning system that classifies YouTube comments as **positive**, **negative**, or **neutral**. The project goes beyond model training — it demonstrates a complete **MLOps workflow** including data versioning, experiment tracking, containerized serving, and automated cloud deployment.

### Key Highlights
- ✅ **LightGBM** classifier with TF-IDF text features
- ✅ **Flask REST API** for real-time inference
- ✅ **Dockerized** for consistent, portable deployment
- ✅ **AWS** cloud deployment (EC2 / ECR)
- ✅ **DVC** for data and model versioning
- ✅ **GitHub Actions** for automated CI/CD
- ✅ **Experiment tracking** for reproducible results

---

## 🏗️ MLOps Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                           │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   Data   │───▶│  Train   │───▶│ Register │───▶│  Serve   │  │
│  │ (DVC)    │    │LightGBM  │    │  Model   │    │  Flask   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│        │               │                               │        │
│        ▼               ▼                               ▼        │
│  ┌──────────┐    ┌──────────┐                   ┌──────────┐   │
│  │  Remote  │    │Experiment│                   │  Docker  │   │
│  │ Storage  │    │Tracking  │                   │Container │   │
│  └──────────┘    └──────────┘                   └──────────┘   │
│                                                        │        │
│                                          ┌─────────────▼──────┐│
│                          GitHub Actions  │   AWS Deployment   ││
│                          (CI/CD)  ──────▶│   (EC2 / ECR)      ││
│                                          └────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **ML Model** | LightGBM, Scikit-learn, TF-IDF |
| **NLP & Text Processing** | NLTK, Regex, Preprocessing Pipeline |
| **API Serving** | Flask, Gunicorn |
| **Containerization** | Docker, Docker Compose |
| **Data Versioning** | DVC (Data Version Control) |
| **Experiment Tracking** | DVC Experiments / MLflow |
| **CI/CD** | GitHub Actions |
| **Cloud Deployment** | AWS EC2, AWS ECR |
| **Version Control** | Git, GitHub |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
Youtube-Sentiment-Insights/
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml           # GitHub Actions CI/CD pipeline
│
├── data/
│   ├── raw/                    # Raw YouTube comments (DVC tracked)
│   └── processed/              # Cleaned & preprocessed data (DVC tracked)
│
├── models/
│   └── sentiment_model.pkl     # Trained LightGBM model (DVC tracked)
│
├── src/
│   ├── data/
│   │   └── preprocess.py       # Text cleaning & feature extraction
│   ├── model/
│   │   ├── train.py            # Model training script
│   │   └── evaluate.py         # Evaluation & metrics
│   └── api/
│       └── app.py              # Flask API application
│
├── dvc.yaml                    # DVC pipeline stages
├── dvc.lock                    # DVC pipeline lock file
├── params.yaml                 # Hyperparameters & config
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Multi-container setup
├── requirements.txt
└── README.md
```

---

## 🔄 ML Pipeline

The pipeline is defined using **DVC** for full reproducibility. Each stage is tracked and cached:

```yaml
# dvc.yaml stages overview
stages:
  preprocess:      # Clean text, remove stopwords, apply TF-IDF
  train:           # Train LightGBM on processed features
  evaluate:        # Compute accuracy, F1, confusion matrix
```

**Run the full pipeline:**
```bash
dvc repro
```

**Track experiments:**
```bash
dvc exp run --set-param model.num_leaves=50
dvc exp show
```

---

## 🌐 API Reference

The Flask API exposes a simple endpoint for real-time sentiment prediction.

**Base URL:** `http://<your-server>:5000`

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

Check API health status.
```json
{ "status": "healthy", "model": "loaded" }
```

---

## ⚙️ CI/CD Pipeline

Automated with **GitHub Actions** — every push to `main` triggers the full pipeline:

```
Push to main
     │
     ▼
┌─────────────────┐
│  1. Run Tests   │  ← pytest unit & integration tests
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Build Docker │  ← docker build
│     Image       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Push to     │  ← Push to AWS ECR
│    AWS ECR      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Deploy to   │  ← SSH into EC2, pull & run new container
│    AWS EC2      │
└─────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- DVC (`pip install dvc`)
- AWS CLI (for cloud deployment)

### 1. Clone the repo
```bash
git clone https://github.com/omarhatem44/Youtube-Sentiment-Insights.git
cd Youtube-Sentiment-Insights
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull data with DVC
```bash
dvc pull
```

### 4. Run the ML pipeline
```bash
dvc repro
```

### 5. Run locally with Docker
```bash
docker-compose up --build
```

The API will be available at `http://localhost:5000`

### 6. Test the API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "This is the best tutorial I have ever watched!"}'
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~XX% |
| **F1-Score (Macro)** | ~XX% |
| **Precision** | ~XX% |
| **Recall** | ~XX% |

> 📝 Replace XX with your actual evaluation numbers from `dvc metrics show`

---

## 🧠 What I Learned (MLOps Skills Demonstrated)

| Skill | Implementation |
|---|---|
| **Data Versioning** | DVC tracks raw data, processed features & model artifacts |
| **Pipeline Reproducibility** | `dvc repro` reruns only changed stages |
| **Experiment Tracking** | Hyperparameter sweeps logged and compared via DVC |
| **Model Serving** | Production Flask API with health checks |
| **Containerization** | Dockerfile + docker-compose for dev & prod parity |
| **CI/CD Automation** | GitHub Actions builds, tests, and deploys on every push |
| **Cloud Deployment** | Docker image pushed to AWS ECR, served on EC2 |

---

## 👤 Author

**Omar Hatem**
- 🎓 Computer Science Student — Modern Academy, Cairo
- 💼 ML & MLOps Engineer
- 🔗 [GitHub](https://github.com/omarhatem44) | [LinkedIn](https://linkedin.com/in/your-profile)

---

<p align="center">
  <i>Built with a focus on production-quality MLOps practices 🚀</i>
</p>
