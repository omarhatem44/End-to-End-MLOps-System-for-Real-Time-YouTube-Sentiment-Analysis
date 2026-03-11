# YouTube Comment Insights 🔍

An end-to-end Machine Learning project that analyzes the sentiment of YouTube video comments in real time.

This project provides a Chrome extension and a web dashboard that automatically fetch comments from any YouTube video and analyzes them using a trained Machine Learning model.

The system visualizes insights such as sentiment distribution, comment trends, and word clouds to help understand audience reactions.

---

# 🚀 Features

• Fetch comments directly from any YouTube video  
• Perform sentiment analysis using a trained ML model  
• Classify comments as:
- Positive
- Neutral
- Negative

• Generate visual insights:
- Sentiment Pie Chart
- Word Cloud
- Sentiment Trend Over Time

• Chrome Extension integration for real-time analysis  
• Flask API for model serving  
• Dockerized application  
• CI/CD pipeline with GitHub Actions  
• Deployment on AWS EC2  

---

# 🧠 Machine Learning Pipeline

The project follows a complete ML workflow:

1️⃣ Data Collection  
- Extract comments using YouTube Data API

2️⃣ Data Preprocessing
- Lowercasing
- Removing special characters
- Stopword removal
- Lemmatization

3️⃣ Feature Engineering
- TF-IDF Vectorization

4️⃣ Model Training
- LightGBM Classifier

5️⃣ Model Evaluation

6️⃣ Model Packaging

7️⃣ Deployment using Flask API

---

# 🏗️ Project Architecture

YouTube Video
↓
Chrome Extension
↓
Flask API
↓
Machine Learning Model
↓
Sentiment Predictions
↓
Charts + Insights


---

# 📂 Project Structure

Youtube-Sentiment-Insights

├── flask_api
│ └── app.py # Flask API for inference
│
├── src
│ ├── data
│ │ ├── data_ingestion.py
│ │ └── data_preprocessing.py
│ │
│ └── model
│ ├── model_building.py
│ ├── model_evaluation.py
│ └── register_model.py
│
├── artifacts
│ ├── lgbm_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── yt-chrome-plugin-frontend
│ ├── manifest.json
│ ├── popup.html
│ └── popup.js
│
├── templates
│ └── dashboard.html
│
├── Dockerfile
├── requirements.txt
├── dvc.yaml
└── README.md


---

# 📊 Dashboard Features

The dashboard provides visual insights including:

• Sentiment Distribution Chart  
• Word Cloud Visualization  
• Comment Sentiment Trends  
• Top Comments with Predictions  

---

# 🧩 Chrome Extension

The Chrome extension automatically:

1. Detects the YouTube video ID  
2. Fetches video comments via YouTube API  
3. Sends comments to the ML API  
4. Displays sentiment insights and visualizations  

---

# 🐳 Docker Deployment

Build Docker Image:
docker build -t youtube-sentiment .

Run Container:
docker run -p 5000:5000 youtube-sentiment

---

# ☁️ AWS Deployment

The project is deployed on an AWS EC2 instance.

Technologies used:

- AWS EC2
- Docker
- GitHub Actions (CI/CD)
- Amazon ECR

Deployment flow:

GitHub Push
↓
GitHub Actions
↓
Build Docker Image
↓
Push to Amazon ECR
↓
Deploy on EC2


---

# 🧪 API Endpoints

