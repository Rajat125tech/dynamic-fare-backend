import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("uber.csv")

print("Columns:", df.columns.tolist())

# Ensure numeric columns
df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
df["surge"] = pd.to_numeric(df["surge"], errors="coerce")
df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce")
df["fare"] = pd.to_numeric(df["fare"], errors="coerce")

df = df.dropna()

FEATURES = [
    "distance",
    "duration",
    "vehicle_type",
    "weather",
    "traffic",
    "surge",
    "hour",
    "is_weekend"
]

X = df[FEATURES]
y = df["fare"]

categorical_features = ["vehicle_type", "weather", "traffic"]
numeric_features = ["distance", "duration", "surge", "hour", "is_weekend"]

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

print("\n----- UBER MODEL PERFORMANCE -----")
print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R2  :", round(r2_score(y_test, y_pred), 4))

# Save model
joblib.dump(model, "../Models/uber_model.pkl")

print("\nUber model saved successfully.")