from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
import os
import time
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ola_model = joblib.load(os.path.join(BASE_DIR, "Models", "ola_model_v3.pkl"))
indrive_model = joblib.load(os.path.join(BASE_DIR, "Models", "indrive_model_v2.pkl"))

class RideRequest(BaseModel):
    distance: float
    duration: float
    vehicle_type: str
    num_passengers: int
    latitude: float
    longitude: float

REPO_ID = "Rajat-10/ride-fare-models"

def load_hf_model(filename):
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        cache_dir="./models"  # saves locally after first download
    )
    return joblib.load(model_path)

rapido_model = None
uber_model = None

@app.on_event("startup")
def load_large_models():
    global rapido_model, uber_model

    print("Loading large models from HuggingFace...")
    rapido_model = load_hf_model("rapido_model.pkl")
    uber_model = load_hf_model("uber_model.pkl")
    print("Large models loaded successfully!")



def fetch_weather(latitude: float, longitude: float):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&current_weather=true"
        )

        response = requests.get(url, timeout=5)
        data = response.json()
        weather_code = data.get("current_weather", {}).get("weathercode", 0)

        # Map Open-Meteo weather codes → model-compatible strings
        if weather_code <= 3:
            return "clear"
        elif 45 <= weather_code <= 48:
            return "fog"
        elif 51 <= weather_code <= 67:
            return "rain"
        elif 71 <= weather_code <= 77:
            return "snow"
        else:
            return "cloudy"

    except Exception:
        return "clear"  # safe fallback
# -----------------------------
# FEATURE GENERATION
# -----------------------------
def generate_features(data: RideRequest):

    now = datetime.now()

    hour = now.hour
    is_weekend = 1 if now.weekday() >= 5 else 0

    if 7 <= hour <= 10 or 17 <= hour <= 20:
        traffic = "high"
    elif 11 <= hour <= 16:
        traffic = "medium"
    else:
        traffic = "low"

    surge = 1.3 if traffic == "high" else 1.0
    weather = fetch_weather(data.latitude, data.longitude)

    print("DEBUG WEATHER:", weather)
    print("DEBUG LAT/LON:", data.latitude, data.longitude)
    
    return {
        "distance": data.distance,
        "duration": data.duration,
        "hour": hour,
        "is_weekend": is_weekend,
        "traffic": traffic,
        "weather": weather,
        "surge": surge,
        "num_passengers": data.num_passengers
    }


# -----------------------------
# VEHICLE MAPPING LAYER
# -----------------------------
def map_vehicle(platform, tier):
    mapping = {
        "rapido": {
            "Budget": "Bike",
            "Standard": "Auto",
            "Premium": "Auto"
        },
        "uber": {
            "Budget": "UberGo",
            "Standard": "UberX",
            "Premium": "UberXL"
        },
        "indrive": {
            "Budget": "Electric",
            "Standard": "Sedan",
            "Premium": "SUV"
        }
    }
    return mapping[platform][tier]


# -----------------------------
# RATE LIMITER
# -----------------------------
RATE_LIMIT = 5
REFILL_TIME = 60
token_buckets = {}
prediction_cache = {}


# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def predict_fare(data: RideRequest, request: Request):

    client_ip = request.client.host
    current_time = time.time()

    if client_ip not in token_buckets:
        token_buckets[client_ip] = {
            "tokens": RATE_LIMIT,
            "last_refill": current_time
        }

    bucket = token_buckets[client_ip]
    elapsed = current_time - bucket["last_refill"]

    if elapsed > REFILL_TIME:
        bucket["tokens"] = RATE_LIMIT
        bucket["last_refill"] = current_time

    if bucket["tokens"] <= 0:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    bucket["tokens"] -= 1

    features = generate_features(data)
    cache_key = tuple(sorted({**features, "tier": data.vehicle_type}.items()))

    if cache_key in prediction_cache:
        cached = prediction_cache[cache_key]
        cached["cache_hit"] = True
        return cached

    # -----------------------------
    # OLA (NYC-based model)
    # -----------------------------
    features_ola = features.copy()
    features_ola["distance"] = features["distance"] * 0.621371  # km → miles

    input_df_ola = pd.DataFrame([features_ola])
    ola_price = ola_model.predict(input_df_ola)[0]

    # USD → INR
    USD_TO_INR = 83
    ola_price *= USD_TO_INR

    # India scaling normalization
    BASE_US_PRICE_PER_KM = 2.2
    BASE_IN_PRICE_PER_KM = 18
    scaling_factor = BASE_IN_PRICE_PER_KM / (BASE_US_PRICE_PER_KM * 83)
    ola_price *= scaling_factor

    # -----------------------------
    # inDrive (NYC-based)
    # -----------------------------
    indrive_vehicle = map_vehicle("indrive", data.vehicle_type)

    features_indrive = features.copy()
    features_indrive["vehicle_type"] = indrive_vehicle

    input_df_indrive = pd.DataFrame([features_indrive])
    indrive_price = indrive_model.predict(input_df_indrive)[0]

    indrive_price *= USD_TO_INR
    indrive_price *= scaling_factor

    # -----------------------------
    # Rapido (India-native)
    # -----------------------------
    rapido_vehicle = map_vehicle("rapido", data.vehicle_type)

    input_df_rapido = pd.DataFrame([{
        "distance": data.distance,
        "surge": features["surge"],
        "vehicle_type": rapido_vehicle,
        "weather": features["weather"],
        "traffic": features["traffic"]
    }])

    rapido_price = rapido_model.predict(input_df_rapido)[0]

    # -----------------------------
    # Uber (India-native)
    # -----------------------------
    uber_vehicle = map_vehicle("uber", data.vehicle_type)

    input_df_uber = pd.DataFrame([{
        "distance": data.distance,
        "surge": features["surge"],
        "vehicle_type": uber_vehicle,
        "weather": features["weather"],
        "traffic": features["traffic"]
    }])

    uber_price = uber_model.predict(input_df_uber)[0]

    response = {
        "ola_price": round(float(ola_price), 2),
        "indrive_price": round(float(indrive_price), 2),
        "rapido_price": round(float(rapido_price), 2),
        "uber_price": round(float(uber_price), 2),
        "computed_hour": features["hour"],
        "computed_traffic": features["traffic"],
        "computed_surge": features["surge"],
        "remaining_tokens": bucket["tokens"],
        "cache_hit": False
    }

    prediction_cache[cache_key] = response

    return response