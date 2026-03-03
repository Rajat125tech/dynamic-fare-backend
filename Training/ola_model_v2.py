import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ola = pd.read_csv("Training/new_ola.csv", low_memory=False)

ola.columns = (
    ola.columns
        .str.replace(r'\+AF8-', '_', regex=True)
        .str.replace('+', '_', regex=False)
        .str.strip()
        .str.lower()
)

ola["pickup_time"] = pd.to_datetime(ola["pickup_time"])
ola["drop_time"] = pd.to_datetime(ola["drop_time"])

ola["duration"] = (ola["drop_time"] - ola["pickup_time"]).dt.total_seconds() / 60

ola["hour"] = ola["pickup_time"].dt.hour

ola["is_weekend"] = ola["pickup_time"].dt.dayofweek >= 5
ola["is_weekend"] = ola["is_weekend"].astype(int)

def traffic_from_hour(h):
    if 7 <= h <= 10 or 17 <= h <= 20:
        return "high"
    elif 11 <= h <= 16:
        return "medium"
    else:
        return "low"

ola["traffic"] = ola["hour"].apply(traffic_from_hour)

ola["weather"] = "clear"
ola["surge"] = np.where(ola["traffic"] == "high", 1.3, 1.0)

ola["distance"] = (
    ola["distance"]
    .astype(str)
    .str.replace(r'[^0-9.]', '', regex=True)
)
ola["distance"] = pd.to_numeric(ola["distance"], errors="coerce")

ola["total_amount"] = (
    ola["total_amount"]
    .astype(str)
    .str.replace(r'[^0-9.]', '', regex=True)
)

ola["total_amount"] = pd.to_numeric(ola["total_amount"], errors="coerce")

ola = ola.dropna(subset=["distance", "duration", "total_amount"])

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

X = ola[FEATURES]
y = ola["total_amount"]

categorical_features = ["traffic", "weather"]
numeric_features = [
    "distance",
    "duration",
    "hour",
    "is_weekend",
    "surge",
    "num_passengers"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


joblib.dump(model, "Models/ola_model_v3.pkl")

print("\n----- OLA (PRE-RIDE) MODEL PERFORMANCE -----")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 4))

#print(ola["total_amount"].describe())