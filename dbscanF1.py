import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv", dtype=str)
df = df.replace('\\N', np.nan)

# Choose numeric features to cluster on (example)
numeric_cols = ['grid', 'laps', 'milliseconds', 'fastestLapSpeed']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=numeric_cols)  # For clustering we drop rows missing cluster features
X = df[numeric_cols].values

# Scale features before clustering
X_scaled = StandardScaler().fit_transform(X)

# Fit DBSCAN - parameters may need tuning for this dataset
db = DBSCAN(eps=0.6, min_samples=8)
labels = db.fit_predict(X_scaled)

# attach labels back for analysis
df_clust = df.copy()
df_clust['dbscan_label'] = labels

# Visualize two principal features
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', s=15)
plt.title("DBSCAN Clusters on F1 features (grid, laps, ...)")
plt.xlabel(numeric_cols[0])
plt.ylabel(numeric_cols[1])
plt.show()

# Print counts per cluster
print(df_clust['dbscan_label'].value_counts())
