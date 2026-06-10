#!/usr/bin/env python
"""Generate RASCAL performance plots using NumPy instead of PyTorch."""

from __future__ import annotations

import io
import json
import os
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


class TorchCheckpointUnpickler(pickle.Unpickler):
    """Minimal PyTorch checkpoint reader for CPU FloatStorage tensors."""

    def __init__(self, handle: io.BytesIO, archive: zipfile.ZipFile, root: str):
        super().__init__(handle)
        self.archive = archive
        self.root = root

    def persistent_load(self, pid: tuple[Any, ...]) -> np.ndarray:
        typename, storage_type, key, location, numel = pid
        if typename != "storage" or storage_type != "FloatStorage" or location != "cpu":
            raise ValueError(f"Unsupported storage reference: {pid!r}")
        raw = self.archive.read(f"{self.root}/data/{key}")
        values = np.frombuffer(raw, dtype="<f4", count=numel).copy()
        return values

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_v2
        if module == "torch" and name == "FloatStorage":
            return "FloatStorage"
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def rebuild_tensor_v2(
    storage: np.ndarray,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: OrderedDict,
) -> np.ndarray:
    item_stride = tuple(int(step) * storage.itemsize for step in stride)
    view = np.lib.stride_tricks.as_strided(
        storage[storage_offset:],
        shape=tuple(int(dim) for dim in size),
        strides=item_stride,
    )
    return np.array(view, copy=True)


def load_checkpoint(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        root = archive.namelist()[0].split("/", 1)[0]
        payload = archive.read(f"{root}/data.pkl")
        return TorchCheckpointUnpickler(io.BytesIO(payload), archive, root).load()


def load_workflow_config() -> dict:
    config_path = os.environ.get("EFT_WORKFLOW_CONFIG")
    if config_path:
        path = Path(config_path)
    else:
        path = REPO_ROOT / "configs" / "paper_10m" / "neural_training.json"
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_storage_workspace(path_config: dict) -> Path:
    workspace = Path(path_config.get("storage_workspace", ".")).expanduser()
    if workspace == Path("."):
        return REPO_ROOT
    if workspace.is_absolute():
        return workspace
    return REPO_ROOT / workspace


WORKFLOW_CONFIG = load_workflow_config()
PATH_CONFIG = WORKFLOW_CONFIG.get("paths", {})
PHYSICS_CONFIG = WORKFLOW_CONFIG.get("physics", {})

STORAGE_WORKSPACE_DIR = resolve_storage_workspace(PATH_CONFIG)
TABLE_DIR = STORAGE_WORKSPACE_DIR / PATH_CONFIG.get("table_subdir", "tables")
INPUT_DIR = TABLE_DIR / PATH_CONFIG.get("sample_output_subdir", "madminer_style_training")
MODEL_DIR = INPUT_DIR / PATH_CONFIG.get("model_subdir", "trained_estimators")
PLOT_DIR = INPUT_DIR / PATH_CONFIG.get("performance_plot_subdir", "performance_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = list(PHYSICS_CONFIG["feature_columns"])
EFT_OPERATORS = list(PHYSICS_CONFIG["eft_operators"])
THETA_COLUMNS = [f"theta0_{name}" for name in EFT_OPERATORS]
SCORE_COLUMNS = [f"score_{name}" for name in EFT_OPERATORS]

CHECKPOINT = load_checkpoint(MODEL_DIR / "rascal.pt")
TRAINING_CONFIG = dict(CHECKPOINT["training_config"])
SCALERS = {
    key: Standardizer(np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))
    for key, value in CHECKPOINT["scalers"].items()
}
STATE = CHECKPOINT["state_dict"]


def ordered_layers() -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    index = 0
    while f"network.{index}.weight" in STATE:
        layers.append((STATE[f"network.{index}.weight"], STATE[f"network.{index}.bias"]))
        index += 2
    return layers


LAYERS = ordered_layers()


def forward_and_score(features: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([features, theta], axis=1).astype(np.float32)
    activations = [x]
    pre_activations = []
    activation_name = str(TRAINING_CONFIG.get("activation", "tanh")).lower()

    hidden_layers = LAYERS[:-1]
    final_weight, final_bias = LAYERS[-1]
    current = x
    for weight, bias in hidden_layers:
        z = current @ weight.T + bias
        pre_activations.append(z)
        if activation_name == "tanh":
            current = np.tanh(z)
        elif activation_name in {"silu", "swish"}:
            current = z / (1.0 + np.exp(-z))
        elif activation_name == "relu":
            current = np.maximum(z, 0.0)
        elif activation_name == "gelu":
            current = 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))
        else:
            raise ValueError(f"Unsupported activation {activation_name!r}")
        activations.append(current)

    scaled_log_r = current @ final_weight.T + final_bias
    log_r = scaled_log_r * SCALERS["log_r"].scale + SCALERS["log_r"].mean

    grad = np.repeat(final_weight, len(x), axis=0).astype(np.float32)
    for layer_index in range(len(hidden_layers) - 1, -1, -1):
        z = pre_activations[layer_index]
        weight, _bias = hidden_layers[layer_index]
        if activation_name == "tanh":
            derivative = 1.0 - np.tanh(z) ** 2
        elif activation_name in {"silu", "swish"}:
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            derivative = sigmoid * (1.0 + z * (1.0 - sigmoid))
        elif activation_name == "relu":
            derivative = (z > 0.0).astype(np.float32)
        elif activation_name == "gelu":
            tanh_arg = np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)
            tanh_val = np.tanh(tanh_arg)
            sech2 = 1.0 - tanh_val**2
            derivative = 0.5 * (1.0 + tanh_val) + 0.5 * z * sech2 * np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * z**2)
        grad = (grad * derivative) @ weight

    theta_start = len(FEATURE_COLUMNS)
    theta_stop = theta_start + len(EFT_OPERATORS)
    score = grad[:, theta_start:theta_stop] * SCALERS["log_r"].scale / SCALERS["theta"].scale
    return log_r.reshape(-1), score


def collect_predictions() -> pd.DataFrame:
    usecols = ["split", "event_id", *FEATURE_COLUMNS, *THETA_COLUMNS, "log_r", *SCORE_COLUMNS]
    frame = pd.read_csv(INPUT_DIR / "ratio_test.csv", usecols=usecols)
    batch_size = int(TRAINING_CONFIG.get("batch_size", 32768))
    rows = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        features = SCALERS["feature"].transform(batch[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        theta = SCALERS["theta"].transform(batch[THETA_COLUMNS].to_numpy(dtype=np.float32))
        log_r_pred, score_pred = forward_and_score(features, theta)
        payload = {
            "method": "RASCAL",
            "event_id": batch["event_id"].to_numpy(),
            "log_r_true": batch["log_r"].to_numpy(dtype=np.float32),
            "log_r_pred": log_r_pred,
        }
        for i, name in enumerate(EFT_OPERATORS):
            payload[f"score_true_{name}"] = batch[f"score_{name}"].to_numpy(dtype=np.float32)
            payload[f"score_pred_{name}"] = score_pred[:, i]
        rows.append(pd.DataFrame(payload))
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(MODEL_DIR / "rascal_test_predictions.csv", index=False)
    return predictions


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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
    plt.figure(figsize=(6, 5))
    plt.scatter(values[true_col], values[pred_col], s=8, alpha=0.35, linewidths=0)
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title("RASCAL")
    plt.grid(True)
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    plt.savefig(PLOT_DIR / f"rascal_{safe_label}_scatter.png", dpi=160)
    plt.close()


def plot_residual_histogram(predictions: pd.DataFrame, true_col: str, pred_col: str, label: str) -> None:
    residual = (predictions[pred_col] - predictions[true_col]).replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(7, 4))
    plt.hist(residual, bins=80, histtype="step", density=True, label="RASCAL")
    plt.xlabel(f"Predicted - true {label}")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    plt.savefig(PLOT_DIR / f"rascal_{safe_label}_residuals.png", dpi=160)
    plt.close()


def main() -> None:
    predictions = collect_predictions()
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
