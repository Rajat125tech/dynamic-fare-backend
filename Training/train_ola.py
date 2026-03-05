# Rajat Srivastava
# OLA Model - Production Stable Version
# Compatible with:
# numpy==1.26.4
# scikit-learn==1.4.2

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("ola.csv", low_memory=False)

# Clean column names
df.columns = (
    df.columns
        .str.replace(r'\+AF8-', '_', regex=True)
        .str.replace('+', '_', regex=False)
        .str.strip()
        .str.lower()
)

print("Cleaned Columns:", df.columns.tolist())


# -----------------------------
# TIME CONVERSION
# -----------------------------
df["pickup_time"] = pd.to_datetime(df["pickup_time"], errors="coerce")
df["drop_time"] = pd.to_datetime(df["drop_time"], errors="coerce")

# Feature engineering
df["duration"] = (df["drop_time"] - df["pickup_time"]).dt.total_seconds() / 60
df["hour"] = df["pickup_time"].dt.hour
df["is_weekend"] = (df["pickup_time"].dt.dayofweek >= 5).astype(int)


# -----------------------------
# TRAFFIC + WEATHER + SURGE
# -----------------------------
def traffic_from_hour(h):
    if 7 <= h <= 10 or 17 <= h <= 20:
        return "high"
    elif 11 <= h <= 16:
        return "medium"
    else:
        return "low"

df["traffic"] = df["hour"].apply(traffic_from_hour)
df["weather"] = "clear"
df["surge"] = np.where(df["traffic"] == "high", 1.3, 1.0)


# -----------------------------
# CLEAN NUMERIC COLUMNS
# -----------------------------
df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

df["total_amount"] = (
    df["total_amount"]
        .astype(str)
        .str.replace(r'\+AC0-', '-', regex=True)
        .str.replace(r'[^0-9\.-]', '', regex=True)
)

df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
df["num_passengers"] = pd.to_numeric(df["num_passengers"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["distance", "duration", "total_amount", "num_passengers"])


# -----------------------------
# DEFINE FEATURES
# -----------------------------
FEATURES = [
    "distance",
    "duration",
    "hour",
    "is_weekend",
    "traffic",
    "weather",
    "surge",
    "num_passengers"
]

X = df[FEATURES].copy()
y = df["total_amount"]

# Explicitly set categorical type (important for stable encoding)
X["traffic"] = X["traffic"].astype("category")
X["weather"] = X["weather"].astype("category")


categorical = ["traffic", "weather"]
numeric = ["distance", "duration", "hour", "is_weekend", "surge", "num_passengers"]


# -----------------------------
# PREPROCESSOR
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ]
)


# -----------------------------
# MODEL PIPELINE
# -----------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])


# -----------------------------
# TRAIN
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))


# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "../Models/ola_model_v3.pkl")

print("✅ OLA model retrained and saved successfully.")