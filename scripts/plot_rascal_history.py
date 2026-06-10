#!/usr/bin/env python
"""Plot the saved RASCAL training history without requiring PyTorch."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "tables" / "madminer_style_training" / "trained_estimators"
PLOT_DIR = REPO_ROOT / "tables" / "madminer_style_training" / "performance_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

history = pd.read_csv(MODEL_DIR / "rascal_history.csv")

plt.figure(figsize=(7, 4))
train = history["train_loss"] / history["train_loss"].iloc[0]
validation = history["validation_loss"] / history["train_loss"].iloc[0]
plt.plot(history["epoch"], train, linestyle="-", label="RASCAL train")
plt.plot(history["epoch"], validation, linestyle="--", label="RASCAL val")
plt.xlabel("Epoch")
plt.ylabel("Loss / initial loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "rascal_training_history.png", dpi=160)
print(f"Wrote {PLOT_DIR / 'rascal_training_history.png'}")
