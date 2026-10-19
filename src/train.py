"""Train and compare regression models for log10(K_VRH) prediction.

Models: Linear Regression, Random Forest, MLP (as proposed), plus
Histogram Gradient Boosting as an extra baseline.
Evaluation: 80/20 hold-out test set + 5-fold CV on the training set.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42

df = pd.read_csv(ROOT / "data" / "features.csv")
X, y = df.drop(columns="log10_kvrh"), df["log10_kvrh"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"Train {len(X_tr)}  Test {len(X_te)}  Features {X.shape[1]}")

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=1, max_features=0.33,
                                           n_jobs=-1, random_state=SEED),
    "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=1e-3,
                                                        learning_rate_init=1e-3, max_iter=500,
                                                        early_stopping=True, random_state=SEED)),
    "Gradient Boosting": HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.03, max_leaf_nodes=31,
                                                       l2_regularization=1.0, random_state=SEED),
}

rows, preds = [], {}
cv = KFold(5, shuffle=True, random_state=SEED)
for name, model in models.items():
    cv_mae = -cross_val_score(model, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    model.fit(X_tr, y_tr)
    p = model.predict(X_te)
    preds[name] = p
    gpa_mae = mean_absolute_error(10 ** y_te, 10 ** p)
    rows.append({
        "Model": name,
        "CV MAE (log10)": f"{cv_mae.mean():.4f} ± {cv_mae.std():.4f}",
        "Test MAE (log10)": round(mean_absolute_error(y_te, p), 4),
        "Test RMSE (log10)": round(np.sqrt(mean_squared_error(y_te, p)), 4),
        "Test R2": round(r2_score(y_te, p), 4),
        "Test MAE (GPa)": round(gpa_mae, 2),
    })
    print(rows[-1])

metrics = pd.DataFrame(rows)
metrics.to_csv(RES / "metrics.csv", index=False)
(RES / "metrics.md").write_text(metrics.to_markdown(index=False))
print("\n" + metrics.to_string(index=False))

# Parity plots
fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4.2))
lo, hi = y.min() - 0.1, y.max() + 0.1
for ax, (name, p) in zip(axes, preds.items()):
    ax.scatter(y_te, p, s=5, alpha=0.4)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="True log10(K_VRH)", ylabel="Predicted",
           title=f"{name}\nR²={r2_score(y_te, p):.3f}  MAE={mean_absolute_error(y_te, p):.3f}")
fig.tight_layout()
fig.savefig(RES / "parity_plots.png", dpi=150)

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(metrics["Model"], metrics["Test MAE (log10)"], color="#4C72B0")
ax.set_ylabel("Test MAE (log10 GPa)")
ax.set_title("Model comparison (lower is better)")
for i, v in enumerate(metrics["Test MAE (log10)"]):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.xticks(rotation=15)
fig.tight_layout()
fig.savefig(RES / "model_comparison.png", dpi=150)

# Feature importance (Random Forest)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.to_csv(RES / "feature_importance.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(7, 6))
imp.head(20)[::-1].plot.barh(ax=ax, color="#55A868")
ax.set_title("Top 20 descriptors (Random Forest importance)")
fig.tight_layout()
fig.savefig(RES / "feature_importance.png", dpi=150)

# Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(y, bins=60, color="#8172B2")
ax.set(xlabel="log10(K_VRH / GPa)", ylabel="Count", title="Target distribution (10,987 materials)")
fig.tight_layout()
fig.savefig(RES / "target_distribution.png", dpi=150)
print("Saved results to", RES)
