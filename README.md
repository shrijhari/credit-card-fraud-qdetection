# Credit-Card Fraud Detection with a Quantum Kernel SVM (UC-016)

BITS F386 project. Use case UC-016 from the QAIC "100 High-Impact Quantum Use Cases" compendium (Amaravati Quantum Valley).

A hybrid quantum-classical classifier: a quantum feature map encodes each transaction into a quantum state,
the kernel is the state overlap `k(x, x') = |<phi(x)|phi(x')>|^2`, and a classical SVM is trained on it.
We compare it with SVM-RBF, logistic regression and random forest. Everything runs on a simulator.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Download `creditcard.csv` from Kaggle into `data/` (see `data/README.md`).

## Run
```bash
python experiments/run_experiments.py --data data/creditcard.csv --exp all --seeds 5   # -> results/E1..E5.csv
python experiments/make_plots.py                                                        # -> figures/*.png
python experiments/make_circuit_figure.py                                               # -> figures/circuit_*.png
```
Single experiments: `--exp E1` ... `--exp E5`.

## Experiments
| ID | What varies |
|----|-------------|
| E1 | Main comparison: QSVM vs SVM-RBF, LogReg, RandomForest (4 qubits, ZZ map, reps 2, 200 per class) |
| E2 | Number of qubits: 4, 6, 8 |
| E3 | Feature map (Z, ZZ linear, ZZ full) and repetitions (1, 2, 3) |
| E4 | Training size: 50, 100, 200 per class |
| E5 | Angle scaling 0.25 to 2.0 (kernel concentration) |

## Layout
- `src/qfraud.py` - data subset, leakage-free preprocessing, exact statevector kernel, models, metrics
- `experiments/` - experiment runner, plots, circuit figure
- `results/`, `figures/` - outputs (commit these for the report)

## Notes
- The kernel is computed exactly (noiseless statevectors). It was checked against Qiskit's `FidelityQuantumKernel`.
- Preprocessing is fitted on training data only. Training uses a balanced subset; say so in the report.
- Tested with qiskit 2.6, qiskit-machine-learning 0.9, scikit-learn 1.9.
