import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("clean_indrive.csv")

print("Columns:", df.columns.tolist())
print("Initial rows:", len(df))

# -----------------------------
# CLEAN NUMERIC COLUMNS
# -----------------------------
numeric_cols = [
    "distance",
    "duration",
    "hour",
    "is_weekend",
    "fare"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["distance", "duration", "fare"])

print("Rows after cleaning:", len(df))

# -----------------------------
# FEATURES
# -----------------------------
FEATURES = [
    "distance",
    "duration",
    "vehicle_type",
    "weather",
    "traffic",
    "surge",          # keep as categorical
    "hour",
    "is_weekend"
]

X = df[FEATURES]
y = df["fare"]

categorical_features = [
    "vehicle_type",
    "weather",
    "traffic",
    "surge"
]

numeric_features = [
    "distance",
    "duration",
    "hour",
    "is_weekend"
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

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\n----- INDRIVE MODEL PERFORMANCE -----")
print("MAE  :", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE :", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R2   :", round(r2_score(y_test, y_pred), 4))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "../Models/indrive_model_v2.pkl")

print("\nInDrive model saved successfully.")