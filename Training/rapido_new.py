import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "rapido.csv"))

df = df[df["status"] == "Completed"]

FEATURES = [
    "distance",
    "surge",
    "vehicle_type",
    "weather",
    "traffic"
]

X = df[FEATURES]
y = df["fare"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_features = ["vehicle_type", "weather", "traffic"]
numeric_features = ["distance", "surge"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\n--- RAPIDO MODEL PERFORMANCE ---")
print("R2:", round(r2_score(y_test, preds), 4))
print("MAE:", round(mean_absolute_error(y_test, preds), 2))

joblib.dump(model, os.path.join(BASE_DIR, "../Models/rapido_model.pkl"))

print("Rapido model saved successfully!")