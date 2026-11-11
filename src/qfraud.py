"""Core library: data, preprocessing, exact quantum kernel, models, metrics."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit.quantum_info import Statevector


def load_data(path):
    return pd.read_csv(path)


def balanced_subset(df, n_per_class, seed):
    fr = df[df.Class == 1].sample(n_per_class, random_state=seed)
    ok = df[df.Class == 0].sample(n_per_class, random_state=seed)
    d = pd.concat([fr, ok]).sample(frac=1, random_state=seed)
    return d.drop(columns=["Class", "Time"]).values, d["Class"].values


def preprocess(Xtr, ytr, Xte, k, scale=1.0):
    """Fit on TRAIN only (no leakage). Output angles in [0, pi*scale]."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    sel = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
    Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
    mm = MinMaxScaler((0, np.pi)).fit(Xtr)
    Xtr, Xte = mm.transform(Xtr), np.clip(mm.transform(Xte), 0, np.pi)
    return Xtr * scale, Xte * scale


def feature_map(kind, k, reps):
    """kind: 'z' | 'zz_linear' | 'zz_full'"""
    if kind == "z":
        return z_feature_map(k, reps=reps)
    ent = "linear" if kind == "zz_linear" else "full"
    return zz_feature_map(k, reps=reps, entanglement=ent)


def _states(fm, X):
    return np.array([Statevector(fm.assign_parameters(x)).data for x in X])


def qkernel(fm, A, B=None):
    """Exact (noiseless) kernel k(x,x') = |<phi(x)|phi(x')>|^2."""
    SA = _states(fm, A)
    SB = SA if B is None else _states(fm, B)
    return np.abs(SA.conj() @ SB.T) ** 2


def metrics(y, pred, score):
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                auc=roc_auc_score(y, score))


def run_once(df, n_per_class=200, k=4, reps=2, fmap="zz_linear", scale=1.0,
             seed=0, baselines=("SVM-RBF", "LogReg", "RandomForest")):
    """One train/test run. Returns a list of result dicts (one per model)."""
    X, y = balanced_subset(df, n_per_class, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    Xtr, Xte = preprocess(Xtr, ytr, Xte, k, scale)
    fm = feature_map(fmap, k, reps)
    Ktr, Kte = qkernel(fm, Xtr), qkernel(fm, Xte, Xtr)
    off = Ktr[~np.eye(len(Ktr), dtype=bool)]
    cfg = dict(n_per_class=n_per_class, k=k, reps=reps, fmap=fmap, scale=scale, seed=seed)
    q = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr)
    rows = [dict(cfg, model="QSVM", kernel_offdiag_mean=off.mean(), kernel_offdiag_std=off.std(),
                 **metrics(yte, q.predict(Kte), q.decision_function(Kte)))]
    models = {"SVM-RBF": SVC(kernel="rbf"),
              "LogReg": LogisticRegression(max_iter=1000),
              "RandomForest": RandomForestClassifier(200, random_state=seed)}
    for name in baselines:
        m = models[name].fit(Xtr, ytr)
        s = m.decision_function(Xte) if hasattr(m, "decision_function") else m.predict_proba(Xte)[:, 1]
        rows.append(dict(cfg, model=name, **metrics(yte, m.predict(Xte), s)))
    return rows
