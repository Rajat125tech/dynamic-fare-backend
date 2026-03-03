import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load("Models/uber_model.pkl")

print("Model Loaded Successfully")

df = pd.read_csv("Training/uber.csv")

X = df.drop("fare", axis=1)  
y = df["fare"]

print("Feature shape:", X.shape)

preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

print("Preprocessor and Model Extracted")

X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

print("Transformed shape:", X_transformed.shape)
print("Feature names:", feature_names)

explainer = shap.TreeExplainer(regressor)

print("SHAP Explainer Created")

sample_indices = np.random.choice(X_transformed.shape[0], 200, replace=False)
X_sample = X_transformed[sample_indices]

shap_values = explainer.shap_values(X_sample)

print("SHAP Values Computed")
shap.summary_plot(shap_values, X_sample, feature_names=feature_names)