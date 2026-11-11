import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
d = pd.read_csv("data/creditcard.csv")
n, nf = len(d), int(d.Class.sum())
print(f"transactions: {n}, frauds: {nf} ({100*nf/n:.3f}%), ratio 1 : {(n-nf)//nf}")
print("missing values:", int(d.isna().sum().sum()))
print("\nAmount by class (0 = genuine, 1 = fraud):")
print(d.groupby("Class").Amount.describe().round(2))
print("\nTime range (hours):", round(d.Time.min() / 3600, 1), "to", round(d.Time.max() / 3600, 1))

X = d.drop(columns=["Class", "Time"])
F, _ = f_classif(X, d.Class)
rank = pd.Series(F, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features by ANOVA F-score (fraud vs genuine):")
print(rank.head(10).round(1))
rank.to_csv("results/feature_ranking_full_data.csv", header=["F_score"])

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
d.Class.value_counts().sort_index().plot(kind="bar", ax=ax[0], logy=True, color=["#2a7de1", "#d9534f"])
ax[0].set_xticklabels(["genuine", "fraud"], rotation=0)
ax[0].set_title("Class counts (log scale)")
for c, l in [(0, "genuine"), (1, "fraud")]:
    ax[1].hist(np.log10(d[d.Class == c].Amount + 1), bins=40, alpha=.6, density=True, label=l)
ax[1].set_xlabel("log10(Amount + 1)"); ax[1].set_title("Amount by class"); ax[1].legend()
rank.head(10).iloc[::-1].plot(kind="barh", ax=ax[2])
ax[2].set_title("Top 10 features (F-score)")
plt.tight_layout(); plt.savefig("figures/data_overview.png", dpi=200)
print("\nsaved figures/data_overview.png")
