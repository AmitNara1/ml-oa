import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Optionally normalize/scale features for models that benefit (not strictly required for tree-based models)
X = df[features].values
y = df['points'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "Bagging": BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=6), n_estimators=15, random_state=42),
    "Boosting": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4), n_estimators=50, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} (F1) - MSE: {mse:.3f}, R2: {r2:.3f}")

    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(6, 3))
        plt.bar(features, model.feature_importances_)
        plt.title(f"{name} - Feature Importances (F1)")
        plt.ylabel("Importance")
        plt.xticks(rotation=45)
        plt.show()
