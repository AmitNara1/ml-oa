import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# Load and clean
df = pd.read_csv("results.csv", dtype=str)
df = df.replace('\\N', np.nan)

numeric_cols = ['grid', 'laps', 'milliseconds', 'fastestLapSpeed', 'number']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df['points'] = pd.to_numeric(df['points'], errors='coerce')
df = df.dropna(subset=['points'])
features = [c for c in numeric_cols if c in df.columns]
df[features] = df[features].fillna(df[features].median())

X = df[features]
y = df['points']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# SHAP explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # regression: returns 2D array

# Summary plot (global)
shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
plt.title("SHAP Summary - F1 Points Prediction")
plt.show()

# Local explanation example: show for one instance as bar plot
idx = 3
sv = shap_values[idx]
plt.barh(features, sv)
plt.title(f"SHAP values for test instance {idx}")
plt.xlabel("SHAP value (impact on predicted points)")
plt.show()
