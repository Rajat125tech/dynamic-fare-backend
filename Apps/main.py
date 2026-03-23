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
    allow_origins=["*",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ola_model = None
indrive_model = None
rapido_model = None
uber_model = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = "Rajat-10/ride-fare-models-2"

def load_local_model(path):
    return joblib.load(path)

def load_hf_model(filename):
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        cache_dir="./models"
    )
    return joblib.load(model_path)

def ensure_models_loaded():
    global ola_model, indrive_model, rapido_model, uber_model
    
    if ola_model is None:
        print("Loading OLA model...")
        ola_model = load_hf_model("ola_model_v3.pkl")

    if indrive_model is None:
        print("Loading inDrive model...")
        indrive_model = load_hf_model("indrive_model_v2.pkl")

    if rapido_model is None:
        print("Loading Rapido model...")
        rapido_model = load_hf_model("rapido_model.pkl")

    if uber_model is None:
        print("Loading Uber model...")
        uber_model = load_hf_model("uber_model.pkl")


class RideRequest(BaseModel):
    distance: float
    duration: float
    vehicle_type: str
    num_passengers: int
    latitude: float
    longitude: float

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

        if weather_code <= 3:
            return "Clear"
        elif 45 <= weather_code <= 48:
            return "Foggy"
        elif 51 <= weather_code <= 67:
            return "Rainy"
        elif 71 <= weather_code <= 77:
            return "Snowy"
        else:
            return "Clear"

    except Exception:
        return "Clear"

def generate_features(data: RideRequest):    #this is the feature engineering part

    now = datetime.now()

    hour = now.hour
    is_weekend = 1 if now.weekday() >= 5 else 0

    if 7 <= hour <= 10 or 17 <= hour <= 20:
        traffic = "High"
    elif 11 <= hour <= 16:
        traffic = "Medium"
    else:
        traffic = "Low"

    surge = 1.3 if traffic == "High" else 1.0
    weather = fetch_weather(data.latitude, data.longitude)

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
    
RATE_LIMIT = 5
REFILL_TIME = 60
token_buckets = {}
prediction_cache = {}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_fare(data: RideRequest, request: Request):

    ensure_models_loaded()

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

    features["distance"] = round(features["distance"], 1) #this will ensure rounding, like 4.7 and then 4.71 hit consider hoga
    features["duration"] = round(features["duration"], 1)

    
    # CACHE KEY
    cache_key = tuple(sorted({**features, "tier": data.vehicle_type}.items()))

    if cache_key in prediction_cache:
        cached = prediction_cache[cache_key]
        cached["cache_hit"] = True
        return cached

    # OLA
    features_ola = features.copy()
    features_ola["distance"] *= 0.621371

    input_df_ola = pd.DataFrame([features_ola])
    ola_price = float(ola_model.predict(input_df_ola)[0])

    USD_TO_INR = 83
    BASE_US_PRICE_PER_KM = 2.2
    BASE_IN_PRICE_PER_KM = 18
    scaling_factor = BASE_IN_PRICE_PER_KM / (BASE_US_PRICE_PER_KM * 83)

    ola_price *= USD_TO_INR * scaling_factor

    # INDRIVE
    indrive_vehicle = map_vehicle("indrive", data.vehicle_type)

    features_indrive = {
        "distance": features["distance"],
        "duration": features["duration"],
        "vehicle_type": indrive_vehicle,
        "weather": features["weather"],
        "traffic": features["traffic"],
        "surge": "Yes" if features["surge"] > 1 else "No",
        "hour": features["hour"],
        "is_weekend": features["is_weekend"]
    }

    input_df_indrive = pd.DataFrame([features_indrive])
    indrive_price = float(indrive_model.predict(input_df_indrive)[0])
    indrive_price *= USD_TO_INR * scaling_factor

    print("inDrive categories:", indrive_model.named_steps["preprocessor"].transformers_[0][1].categories_)
    # RAPIDO
    rapido_vehicle = map_vehicle("rapido", data.vehicle_type)

    input_df_rapido = pd.DataFrame([{
        "distance": features["distance"],
        "duration": features["duration"],
        "vehicle_type": rapido_vehicle,
        "weather": features["weather"],
        "traffic": features["traffic"],
        "surge": features["surge"],
        "hour": features["hour"],
        "is_weekend": features["is_weekend"]
    }])

    rapido_price = float(rapido_model.predict(input_df_rapido)[0])

    # UBER
    uber_vehicle = map_vehicle("uber", data.vehicle_type)

    input_df_uber = pd.DataFrame([{
        "distance": features["distance"],
        "duration": features["duration"],
        "vehicle_type": uber_vehicle,
        "weather": features["weather"],
        "traffic": features["traffic"],
        "surge": features["surge"],
        "hour": features["hour"],
        "is_weekend": features["is_weekend"]
    }])

    uber_price = float(uber_model.predict(input_df_uber)[0])

    response = {
        "ola_price": round(ola_price, 2),
        "indrive_price": round(indrive_price, 2),
        "rapido_price": round(rapido_price, 2),
        "uber_price": round(uber_price, 2),
        "computed_hour": features["hour"],
        "computed_traffic": features["traffic"],
        "computed_surge": features["surge"],
        "remaining_tokens": bucket["tokens"],
        "cache_hit": False
    }

    prediction_cache[cache_key] = response

    return response
