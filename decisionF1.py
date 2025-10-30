import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ----- Load the F1 results.csv -----
df = pd.read_csv("results.csv", dtype=str)  # load as str to handle '\N'

# ----- Basic cleaning -----
df = df.replace('\\N', np.nan)  # replace backslash-N with NaN

# Convert columns to numeric where appropriate
numeric_cols = ['grid', 'laps', 'milliseconds', 'fastestLapSpeed', 'number']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Choose features: numeric ones that are meaningful and available
features = [c for c in numeric_cols if c in df.columns]
# Remove rows without target 'points'
df['points'] = pd.to_numeric(df['points'], errors='coerce')
df = df.dropna(subset=['points'])

# Fill missing numeric features with median (simple robust imputation)
df[features] = df[features].fillna(df[features].median())

# Define X and y
X = df[features].values
y = df['points'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42, max_depth=8)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Decision Tree (F1) - MSE: {mse:.3f}, R2: {r2:.3f}")

# Plot feature importances and tree
plt.figure(figsize=(8, 4))
plt.bar(features, model.feature_importances_)
plt.title("Feature Importances - Decision Tree")
plt.ylabel("Importance")
plt.show()

plt.figure(figsize=(12, 6))
plot_tree(model, filled=True, feature_names=features, max_depth=3)
plt.title("Decision Tree (truncated depth=3) - F1 data")
plt.show()
