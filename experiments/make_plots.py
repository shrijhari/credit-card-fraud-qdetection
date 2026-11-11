"""Turn results/*.csv into figures/*.png.  Usage: python experiments/make_plots.py"""
import os, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results", "figures"
os.makedirs(F, exist_ok=True)

def line(csv, x, title, fname, xlabel=None, metric="f1"):
    p = os.path.join(R, csv)
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    g = d.groupby([x, "model"])[metric].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(5, 3.4))
    for m, s in g.groupby("model"):
        plt.errorbar(s[x], s["mean"], yerr=s["std"], marker="o", capsize=3, label=m)
    plt.xlabel(xlabel or x); plt.ylabel(metric.upper()); plt.title(title)
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(F, fname), dpi=200); plt.close()

# E1: bar chart of all models
p = os.path.join(R, "E1.csv")
if os.path.exists(p):
    d = pd.read_csv(p).groupby("model")[["precision", "recall", "f1", "auc"]].agg(["mean", "std"])
    d.round(3).to_csv(os.path.join(R, "E1_summary.csv"))
    m = d.xs("mean", axis=1, level=1); s = d.xs("std", axis=1, level=1)
    m.plot(kind="bar", yerr=s, figsize=(6, 3.6), capsize=2)
    plt.ylabel("score"); plt.title("E1: QSVM vs classical (mean +/- std, 5 splits)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    plt.xticks(rotation=0); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(F, "E1_models.png"), dpi=200, bbox_inches="tight"); plt.close()

line("E2.csv", "k", "E2: number of qubits", "E2_qubits.png", "qubits (features)")
line("E4.csv", "n_per_class", "E4: training size", "E4_size.png", "samples per class")
line("E5.csv", "scale", "E5: angle scaling", "E5_scale.png", "angle scale factor")

# E3: grouped by feature map and reps (QSVM only)
p = os.path.join(R, "E3.csv")
if os.path.exists(p):
    d = pd.read_csv(p); d = d[d.model == "QSVM"]
    g = d.groupby(["fmap", "reps"])["f1"].agg(["mean", "std"]).reset_index()
    pv, ps = g.pivot(index="reps", columns="fmap", values="mean"), g.pivot(index="reps", columns="fmap", values="std")
    pv.plot(kind="bar", yerr=ps, figsize=(5.5, 3.4), capsize=2)
    plt.ylabel("F1"); plt.title("E3: feature map and repetitions"); plt.xticks(rotation=0)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    plt.grid(axis="y", alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(F, "E3_circuit.png"), dpi=200, bbox_inches="tight"); plt.close()

# E5: kernel concentration (off-diagonal kernel mean/std vs scale)
p = os.path.join(R, "E5.csv")
if os.path.exists(p):
    d = pd.read_csv(p); d = d[d.model == "QSVM"].groupby("scale")[["kernel_offdiag_mean", "kernel_offdiag_std"]].mean()
    plt.figure(figsize=(5, 3.4))
    plt.plot(d.index, d.kernel_offdiag_mean, "o-", label="mean off-diagonal k")
    plt.plot(d.index, d.kernel_offdiag_std, "s--", label="std off-diagonal k")
    plt.xlabel("angle scale factor"); plt.title("E5: kernel concentration")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(F, "E5_kernel_concentration.png"), dpi=200); plt.close()
print("figures written to", F)
