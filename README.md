# 🚖 Dynamic Multi-Platform Ride Fare Prediction System

A full-stack machine learning project that predicts and compares ride fares across multiple ride-hailing platforms including **Ola, Uber, Rapido, and inDrive**.

The system uses trained ML models to estimate ride prices dynamically based on ride parameters, traffic conditions, time of day, and weather data.

---

## 🔥 Features

- Predicts fares for:
  - Ola
  - Uber
  - Rapido
  - inDrive
- Real-time feature engineering:
  - Hour-based traffic logic
  - Weekend detection
  - Surge multiplier calculation
  - Weather-based adjustment using latitude & longitude
- Platform-specific vehicle type mapping
- REST API built using FastAPI
- Rate limiting and caching implemented
- Large models hosted on HuggingFace Hub
- Frontend integrated and deployment-ready

---

## 🧠 Machine Learning

- Built 4 independent regression pipelines using:
  - `scikit-learn`
  - `ColumnTransformer`
  - `OneHotEncoder(handle_unknown="ignore")`
- Feature set includes:
  - Distance
  - Duration
  - Hour
  - Weekend flag
  - Traffic level
  - Weather
  - Surge multiplier
  - Passenger count (OLA)

Models are serialized using `joblib` and loaded dynamically.

---

## ⚙️ Backend

- Built using **FastAPI**
- Lazy model loading
- IP-based rate limiting (token bucket)
- Prediction caching
- Weather API integration (Open-Meteo)
- HuggingFace model download integration

---

## 🌐 Frontend

- Interactive ride comparison UI
- Displays predicted prices from all platforms
- Session-based result handling
- Deployment-ready configuration

---

## 🛠 Tech Stack

- Python 3.11
- FastAPI
- scikit-learn
- NumPy
- Pandas
- Joblib
- HuggingFace Hub
- HTML / CSS / JavaScript

---

## 🚀 Deployment

- Backend: Render
- Frontend: Vercel
- Model Storage: HuggingFace Hub

---

## 📌 How It Works

1. User enters ride details.
2. Backend generates dynamic features.
3. Models predict platform-specific fares.
4. Results are returned and displayed on the dashboard.

---

## 📈 Future Improvements

- Add deep learning models
- Add real-time traffic API integration
- Add historical ride analytics dashboard
- Optimize inference latency

---

Built as a production-oriented ML deployment project.
