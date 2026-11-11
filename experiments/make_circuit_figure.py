"""Save the feature-map circuit diagram (for the report). Usage: python experiments/make_circuit_figure.py"""
import os, matplotlib
matplotlib.use("Agg")
from qiskit.circuit.library import zz_feature_map
os.makedirs("figures", exist_ok=True)
fm = zz_feature_map(4, reps=1, entanglement="linear")
fm.decompose().draw("mpl", filename="figures/circuit_zz_4q_reps1.png", fold=-1)
print("saved figures/circuit_zz_4q_reps1.png")
