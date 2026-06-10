#!/usr/bin/env python
"""Generate performance plots for the saved RASCAL estimator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


def load_workflow_config() -> dict:
    config_path = os.environ.get("EFT_WORKFLOW_CONFIG")
    if not config_path:
        config_path = REPO_ROOT / "configs" / "paper_10m" / "neural_training.json"
    with open(config_path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


WORKFLOW_CONFIG = load_workflow_config()
PATH_CONFIG = WORKFLOW_CONFIG.get("paths", {})
PHYSICS_CONFIG = WORKFLOW_CONFIG.get("physics", {})

STORAGE_WORKSPACE_DIR = Path(PATH_CONFIG.get("storage_workspace", ".")).expanduser()
if STORAGE_WORKSPACE_DIR == Path("."):
    STORAGE_WORKSPACE_DIR = REPO_ROOT
elif not STORAGE_WORKSPACE_DIR.is_absolute():
    STORAGE_WORKSPACE_DIR = REPO_ROOT / STORAGE_WORKSPACE_DIR
TABLE_DIR = STORAGE_WORKSPACE_DIR / PATH_CONFIG.get("table_subdir", "tables")
INPUT_DIR = TABLE_DIR / PATH_CONFIG.get("sample_output_subdir", "madminer_style_training")
MODEL_DIR = INPUT_DIR / PATH_CONFIG.get("model_subdir", "trained_estimators")
PLOT_DIR = INPUT_DIR / PATH_CONFIG.get("performance_plot_subdir", "performance_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = list(PHYSICS_CONFIG["feature_columns"])
EFT_OPERATORS = list(PHYSICS_CONFIG["eft_operators"])
THETA_COLUMNS = [f"theta0_{name}" for name in EFT_OPERATORS]
SCORE_COLUMNS = [f"score_{name}" for name in EFT_OPERATORS]


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


CHECKPOINT = load_checkpoint(MODEL_DIR / "rascal.pt")
TRAINING_CONFIG = dict(CHECKPOINT["training_config"])
SCALERS = {
    key: Standardizer(np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))
    for key, value in CHECKPOINT["scalers"].items()
}


def activation_layer() -> nn.Module:
    activation = str(TRAINING_CONFIG.get("activation", "tanh")).lower()
    if activation == "tanh":
        return nn.Tanh()
    if activation in {"silu", "swish"}:
        return nn.SiLU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation function: {activation!r}")


def build_mlp(input_dim: int, output_dim: int, hidden_layers: Iterable[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    dropout = float(TRAINING_CONFIG.get("dropout", 0.0))
    for width in hidden_layers:
        layers.extend([nn.Linear(previous, int(width)), activation_layer()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class RatioEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Iterable[int]):
        super().__init__()
        self.network = build_mlp(input_dim, 1, hidden_layers)

    def forward(self, features: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        scaled = self.network(torch.cat([features, theta], dim=1))
        mean = torch.as_tensor(SCALERS["log_r"].mean, dtype=torch.float32, device=scaled.device)
        scale = torch.as_tensor(SCALERS["log_r"].scale, dtype=torch.float32, device=scaled.device)
        return scaled * scale + mean


def ratio_score_from_gradient(
    model: RatioEstimator, features: torch.Tensor, theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = theta.detach().clone().requires_grad_(True)
    log_r = model(features, theta)
    grad_scaled = torch.autograd.grad(log_r.sum(), theta, create_graph=False)[0]
    theta_scale = torch.as_tensor(SCALERS["theta"].scale, dtype=torch.float32, device=features.device)
    return log_r, grad_scaled / theta_scale


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_model() -> RatioEstimator:
    input_dim = len(FEATURE_COLUMNS) + len(EFT_OPERATORS)
    model = RatioEstimator(input_dim, TRAINING_CONFIG["hidden_layers"]).to(DEVICE)
    model.load_state_dict(CHECKPOINT["state_dict"])
    model.eval()
    return model


def collect_predictions(model: RatioEstimator) -> pd.DataFrame:
    usecols = ["split", "event_id", *FEATURE_COLUMNS, *THETA_COLUMNS, "log_r", *SCORE_COLUMNS]
    frame = pd.read_csv(INPUT_DIR / "ratio_test.csv", usecols=usecols)
    batch_size = int(TRAINING_CONFIG.get("batch_size", 32768))
    rows = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        features = torch.as_tensor(
            SCALERS["feature"].transform(batch[FEATURE_COLUMNS].to_numpy(dtype=np.float32)),
            dtype=torch.float32,
            device=DEVICE,
        )
        theta = torch.as_tensor(
            SCALERS["theta"].transform(batch[THETA_COLUMNS].to_numpy(dtype=np.float32)),
            dtype=torch.float32,
            device=DEVICE,
        )
        log_r_pred, score_pred = ratio_score_from_gradient(model, features, theta)
        payload = {
            "method": "RASCAL",
            "event_id": batch["event_id"].to_numpy(),
            "log_r_true": batch["log_r"].to_numpy(dtype=np.float32),
            "log_r_pred": log_r_pred.detach().cpu().numpy().ravel(),
        }
        for i, name in enumerate(EFT_OPERATORS):
            payload[f"score_true_{name}"] = batch[f"score_{name}"].to_numpy(dtype=np.float32)
            payload[f"score_pred_{name}"] = score_pred[:, i].detach().cpu().numpy()
        rows.append(pd.DataFrame(payload))
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(MODEL_DIR / "rascal_test_predictions.csv", index=False)
    return predictions


def write_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = [("log_r", "log_r_true", "log_r_pred")]
    targets.extend((f"score_{name}", f"score_true_{name}", f"score_pred_{name}") for name in EFT_OPERATORS)
    for target, true_col, pred_col in targets:
        values = predictions[[true_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        residual = values[pred_col] - values[true_col]
        rows.append(
            {
                "method": "RASCAL",
                "target": target,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "corr": pearson_corr(values[true_col].to_numpy(), values[pred_col].to_numpy()),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(MODEL_DIR / "rascal_test_metrics.csv", index=False)
    return metrics


def plot_history() -> None:
    history_path = MODEL_DIR / "rascal_history.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    plt.figure()
    train = history["train_loss"] / history["train_loss"].iloc[0]
    validation = history["validation_loss"] / history["train_loss"].iloc[0]
    plt.plot(history["epoch"], train, linestyle="-", label="RASCAL train")
    plt.plot(history["epoch"], validation, linestyle="--", label="RASCAL val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / initial loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "rascal_training_history.png", dpi=160)
    plt.close()


def plot_prediction_scatter(predictions: pd.DataFrame, true_col: str, pred_col: str, label: str) -> None:
    values = predictions[[true_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return
    max_points = int(TRAINING_CONFIG.get("plot_max_points", len(values)))
    if len(values) > max_points:
        values = values.sample(max_points, random_state=int(TRAINING_CONFIG["seed"]))
    low = float(np.percentile(values.to_numpy().ravel(), 1))
    high = float(np.percentile(values.to_numpy().ravel(), 99))
    if np.isclose(low, high):
        low, high = float(values.min().min()), float(values.max().max())
    plt.figure()
    plt.scatter(values[true_col], values[pred_col], s=8, alpha=0.35, linewidths=0)
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title("RASCAL")
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    plt.savefig(PLOT_DIR / f"rascal_{safe_label}_scatter.png", dpi=160)
    plt.close()


def plot_residual_histogram(predictions: pd.DataFrame, true_col: str, pred_col: str, label: str) -> None:
    residual = (predictions[pred_col] - predictions[true_col]).replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure()
    plt.hist(residual, bins=80, histtype="step", density=True, label="RASCAL")
    plt.xlabel(f"Predicted - true {label}")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    plt.savefig(PLOT_DIR / f"rascal_{safe_label}_residuals.png", dpi=160)
    plt.close()


def main() -> None:
    model = load_model()
    predictions = collect_predictions(model)
    metrics = write_metrics(predictions)
    plot_history()
    plot_prediction_scatter(predictions, "log_r_true", "log_r_pred", "log_r")
    plot_residual_histogram(predictions, "log_r_true", "log_r_pred", "log_r")
    for name in EFT_OPERATORS:
        plot_prediction_scatter(predictions, f"score_true_{name}", f"score_pred_{name}", f"score_{name}")
        plot_residual_histogram(predictions, f"score_true_{name}", f"score_pred_{name}", f"score_{name}")
    print(metrics.to_string(index=False))
    print(f"Wrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
