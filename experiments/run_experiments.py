"""Run experiments E1-E5. Usage:
    python experiments/run_experiments.py --data data/creditcard.csv --exp all --seeds 5
Results are written to results/<EXP>.csv
"""
import argparse, itertools, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qfraud import load_data, run_once, run_tuned

BOTH = ("SVM-RBF",)                     # light baseline for sweeps
ALL = ("SVM-RBF", "LogReg", "RandomForest")

EXPS = {  # name -> (list of config dicts, baselines)
    "E1": ([dict()], ALL),                                              # main comparison
    "E2": ([dict(k=k) for k in (4, 6, 8)], BOTH),                       # qubits
    "E3": ([dict(fmap=f, reps=r) for f in ("z", "zz_linear", "zz_full")
            for r in (1, 2, 3)], BOTH),                                 # circuit design
    "E4": ([dict(n_per_class=n) for n in (50, 100, 200)], BOTH),        # training size
    "E5": ([dict(scale=s) for s in (0.25, 0.5, 1.0, 2.0)], BOTH),       # data scaling
}
# E8: tuned comparison (hyperparameters chosen by cross-validation on training data only)
E8_CONFIGS = [dict(k=k, fmap=f) for k in (4, 6, 8) for f in ("z", "zz_linear")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--exp", default="all", help="E1..E5, E8 or all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_data(a.data)
    names = list(EXPS) + ["E8"] if a.exp == "all" else [a.exp]
    for name in names:
        if name == "E8":
            rows = []
            for cfg, seed in itertools.product(E8_CONFIGS, range(a.seeds)):
                rows += [dict(exp="E8", **r) for r in run_tuned(df, seed=seed, **cfg)]
                print("E8", cfg, "seed", seed, flush=True)
            pd.DataFrame(rows).to_csv(f"{a.out}/E8.csv", index=False)
            print("saved", f"{a.out}/E8.csv")
            continue
        cfgs, base = EXPS[name]
        rows = []
        for cfg, seed in itertools.product(cfgs, range(a.seeds)):
            rows += [dict(exp=name, **r) for r in run_once(df, seed=seed, baselines=base, **cfg)]
            print(name, cfg, "seed", seed, flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/{name}.csv", index=False)
        print("saved", f"{a.out}/{name}.csv")

if __name__ == "__main__":
    main()
