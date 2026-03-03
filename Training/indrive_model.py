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
indrive = pd.read_csv("Training/clean_indrive.csv")

# -----------------------------
# CLEAN is_weekend (Yes/No → 1/0)
# -----------------------------
# -----------------------------
# CLEAN surge (Yes/No → 1.3 / 1.0)
# -----------------------------
indrive["surge"] = indrive["surge"].replace({
    "Yes": 1.3,
    "No": 1.0
})

indrive["surge"] = pd.to_numeric(indrive["surge"], errors="coerce")

# Force numeric columns to proper numeric type
numeric_cols = ["distance", "duration", "hour", "is_weekend", "surge"]

for col in numeric_cols:
    indrive[col] = pd.to_numeric(indrive[col], errors="coerce")

indrive = indrive.dropna(subset=numeric_cols + ["fare"])

#indrive["is_weekend"] = indrive["is_weekend"].astype(int)

# -----------------------------
# FINAL FEATURE SET
# -----------------------------
FEATURES = [
    "distance",
    "duration",
    "hour",
    "is_weekend",
    "traffic",
    "weather",
    "surge",
    "vehicle_type"
]

X = indrive[FEATURES]
y = indrive["fare"]

# -----------------------------
# PREPROCESSOR
# -----------------------------
categorical_features = ["traffic", "weather", "vehicle_type"]
numeric_features = [
    "distance",
    "duration",
    "hour",
    "is_weekend",
    "surge"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# -----------------------------
# MODEL PIPELINE
# -----------------------------
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
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "Models/indrive_model_v2.pkl")

print("\n----- inDrive (PRE-RIDE) MODEL PERFORMANCE -----")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 4))

