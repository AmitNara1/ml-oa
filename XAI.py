import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----- Replace with your dataset later -----
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
# -------------------------------------------
feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

# ---- SHAP Explanation ----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[0]
else:
    shap_values_to_plot = shap_values

shap.summary_plot(shap_values_to_plot, X_test, show=False)
plt.title("SHAP Summary Plot - Feature Importance")
plt.show()

# ---- LIME Explanation ----
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)

# Explain one test instance
i = 5
exp = explainer_lime.explain_instance(X_test.iloc[i], model.predict_proba)
exp.show_in_notebook(show_table=True, show_all=False)
exp.as_pyplot_figure()
plt.title("LIME Local Explanation")
plt.show()
