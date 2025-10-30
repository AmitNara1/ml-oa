import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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

X = df[features].values
y = df['points'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor(n_neighbors=7)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"KNN (F1) - MSE: {mse:.3f}, R2: {r2:.3f}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual points")
plt.ylabel("Predicted points")
plt.title("KNN Regressor (F1) - Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()
