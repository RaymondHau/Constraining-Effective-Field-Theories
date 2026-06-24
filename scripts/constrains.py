#!/usr/bin/env python3
"""Build neural likelihood-ratio constraint heatmaps.

This stage assumes validation events have already been converted to feature
arrays by ``validation_events.py``.  It loads trained ratio-estimator
checkpoints, evaluates summed log-ratios on a Wilson-coefficient grid, applies
a parameter-binned residual calibration from the held-out prediction table, and
writes the tables, diagnostics, and error-aware q heatmaps used for the final
constraints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESS_NAME = os.environ.get("EFT_PROCESS", "WBF")
PROCESS_CONFIG_DIR = PROJECT_DIR / "configs" / PROCESS_NAME
DEFAULT_CONFIG_PATH = PROCESS_CONFIG_DIR / "constrains_config.json"
DEFAULT_VALIDATION_CONFIG_PATH = PROCESS_CONFIG_DIR / "validation_events_config.json"
DEFAULT_PREDICTION_TABLE_PATH = (
    PROJECT_DIR / "table_outputs" / "madminer_style_training" / "trained_estimators" / "test_predictions.csv"
)


@dataclass(frozen=True)
class ThetaPoint:
    c1: float
    c2: float

    @classmethod
    def from_value(cls, value: Any) -> "ThetaPoint":
        if isinstance(value, Mapping):
            return cls(float(value["c1"]), float(value["c2"]))
        if isinstance(value, Sequence) and len(value) == 2:
            return cls(float(value[0]), float(value[1]))
        raise ValueError(f"Invalid theta point {value!r}; expected [c1, c2] or c1/c2 dict.")

    @property
    def tag(self) -> str:
        text = f"c1_{self.c1:+.6g}_c2_{self.c2:+.6g}"
        return text.replace("+", "p").replace("-", "m").replace(".", "p")

    def as_array(self) -> np.ndarray:
        return np.array([self.c1, self.c2], dtype=np.float32)


@dataclass(frozen=True)
class ParameterCalibration:
    c1_edges: np.ndarray
    c2_edges: np.ndarray
    bias: np.ndarray
    sigma: np.ndarray
    counts: np.ndarray
    sigma_floor: float
    global_bias: float
    global_sigma: float
    apply_bias_correction: bool = False

    def lookup_many(self, points: Sequence[ThetaPoint]) -> tuple[np.ndarray, np.ndarray]:
        c1 = np.array([point.c1 for point in points], dtype=float)
        c2 = np.array([point.c2 for point in points], dtype=float)
        bias = interpolate_parameter_grid(c1, c2, self.c1_edges, self.c2_edges, self.bias)
        sigma = interpolate_parameter_grid(c1, c2, self.c1_edges, self.c2_edges, self.sigma)
        if not self.apply_bias_correction:
            bias = np.zeros_like(bias)
        sigma = np.maximum(sigma, self.sigma_floor)
        return bias.astype(np.float32), sigma.astype(np.float32)


@dataclass(frozen=True)
class ScoreCalibration:
    """Held-out mean and covariance of the learned marginal score."""

    theta: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    counts: np.ndarray
    coordinate_scale: np.ndarray
    neighbors: int = 4
    distance_power: float = 2.0
    covariance_ridge_fraction: float = 1.0e-4
    include_interpolation_uncertainty: bool = True

    def lookup_many(self, points: Sequence[ThetaPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return score mean, event covariance, and covariance of the estimated mean."""

        query = np.stack([point.as_array() for point in points]).astype(float)
        scale = np.maximum(np.asarray(self.coordinate_scale, dtype=float), 1.0e-12)
        distances = np.linalg.norm(
            (query[:, None, :] - np.asarray(self.theta, dtype=float)[None, :, :]) / scale[None, None, :],
            axis=2,
        )
        n_neighbors = max(1, min(int(self.neighbors), len(self.theta)))
        nearest = np.argpartition(distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
        means = np.empty((len(points), self.mean.shape[1]), dtype=float)
        covariances = np.empty((len(points), self.covariance.shape[1], self.covariance.shape[2]), dtype=float)
        mean_covariances = np.empty_like(covariances)
        identity = np.eye(self.covariance.shape[1], dtype=float)

        for row, indices in enumerate(nearest):
            selected_distances = distances[row, indices]
            exact = selected_distances <= 1.0e-12
            if exact.any():
                weights = exact.astype(float)
            else:
                weights = np.power(np.maximum(selected_distances, 1.0e-12), -float(self.distance_power))
            weights /= weights.sum()
            means[row] = np.tensordot(weights, self.mean[indices], axes=1)
            covariance = np.tensordot(weights, self.covariance[indices], axes=1)
            variance_scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0e-12)
            covariances[row] = covariance + float(self.covariance_ridge_fraction) * variance_scale * identity
            # Independent calibration-group means contribute w_i^2 V_i / n_i.
            mean_covariance = np.tensordot(
                np.square(weights) / np.maximum(self.counts[indices], 1),
                self.covariance[indices],
                axes=1,
            )
            if self.include_interpolation_uncertainty and not exact.any():
                differences = self.mean[indices] - means[row]
                mean_covariance += np.einsum("i,ij,ik->jk", weights, differences, differences)
            mean_covariances[row] = mean_covariance + float(self.covariance_ridge_fraction) * variance_scale * identity
        return means, covariances, mean_covariances


def interpolate_parameter_grid(
    c1: np.ndarray,
    c2: np.ndarray,
    c1_edges: np.ndarray,
    c2_edges: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    c1_centers = 0.5 * (c1_edges[:-1] + c1_edges[1:])
    c2_centers = 0.5 * (c2_edges[:-1] + c2_edges[1:])
    c1_pos = np.interp(c1, c1_centers, np.arange(len(c1_centers), dtype=float))
    c2_pos = np.interp(c2, c2_centers, np.arange(len(c2_centers), dtype=float))
    c1_low = np.floor(c1_pos).astype(int)
    c2_low = np.floor(c2_pos).astype(int)
    c1_high = np.clip(c1_low + 1, 0, len(c1_centers) - 1)
    c2_high = np.clip(c2_low + 1, 0, len(c2_centers) - 1)
    c1_low = np.clip(c1_low, 0, len(c1_centers) - 1)
    c2_low = np.clip(c2_low, 0, len(c2_centers) - 1)
    wx = c1_pos - c1_low
    wy = c2_pos - c2_low
    v00 = values[c1_low, c2_low]
    v10 = values[c1_high, c2_low]
    v01 = values[c1_low, c2_high]
    v11 = values[c1_high, c2_high]
    return (1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10 + (1.0 - wx) * wy * v01 + wx * wy * v11


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    config["_project_dir"] = str(PROJECT_DIR)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(config["_project_dir"]) / path


def load_validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(config, config.get("validation_config", DEFAULT_VALIDATION_CONFIG_PATH))
    with path.open("r", encoding="utf-8-sig") as handle:
        validation = json.load(handle)
    validation["_project_dir"] = config["_project_dir"]
    return validation


def resolve_manifest_event_path(value: str | Path, output_dir: Path, theta_tag: str) -> Path:
    """Resolve a validation-manifest path, rebasing stale absolute paths.

    Older manifests stored absolute paths from the machine that generated the
    events.  When such a manifest is copied to another machine, use the same
    theta directory beneath the locally configured validation output instead.
    """

    path = Path(value).expanduser()
    text = str(value)
    is_absolute = path.is_absolute() or PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()
    if not is_absolute or path.exists():
        return path
    return output_dir / theta_tag / path.name


def coefficient_names(config: Mapping[str, Any]) -> list[str]:
    if config.get("coefficient_names"):
        return list(config["coefficient_names"])
    try:
        return list(load_validation_config(config).get("coefficient_names", ["c1", "c2"]))
    except OSError:
        return ["c1", "c2"]


def validation_datasets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not config.get("datasets_from_validation", True):
        return list(config["datasets"])

    validation = load_validation_config(config)
    output_dir = resolve_path(config, validation.get("output_dir", "validation_events"))
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            manifest = json.load(handle)
        datasets = []
        for index, item in enumerate(manifest.get("datasets", [])):
            theta_tag = item.get("theta_tag", f"validation_{index:03d}")
            event_file = item.get("feature_file") or item["event_file"]
            datasets.append(
                {
                    "name": theta_tag,
                    "theta_true": item["theta_true"],
                    "event_file": str(resolve_manifest_event_path(event_file, output_dir, theta_tag)),
                }
            )
        return datasets

    event_filename = config.get("validation_event_filename", "features.npy")
    return [
        {
            "name": theta.tag,
            "theta_true": [theta.c1, theta.c2],
            "event_file": str(output_dir / theta.tag / event_filename),
        }
        for theta in map(ThetaPoint.from_value, validation.get("theta_true", []))
    ]


def load_feature_array(path: str | Path, config: Mapping[str, Any], max_events: int | None) -> np.ndarray:
    path = resolve_path(config, path)
    suffixes = path.suffixes
    feature_config = config.get("feature_loading", {})

    if path.suffix == ".npy":
        events = np.load(path)
    elif path.suffix == ".npz":
        events = np.load(path)[feature_config.get("npz_key", "features")]
    elif path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
        columns = feature_config.get("feature_columns")
        events = frame[columns].to_numpy() if columns else frame.to_numpy()
    else:
        raise ValueError(
            f"{path} is not a prepared feature array. Run validation_events.py first "
            "or point the dataset at .npy, .npz, or .csv features."
        )

    events = np.asarray(events, dtype=np.float32)
    if events.ndim != 2:
        raise ValueError(f"Expected a 2D feature array in {path}, got shape {events.shape}.")
    if not np.isfinite(events).all():
        raise ValueError(f"Feature array {path} contains NaN or infinite values.")
    return events[:max_events] if max_events is not None else events


def activation_layer(name: str) -> Any:
    from torch import nn

    activations = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation function {name!r}.")
    return activations[name]()


def build_mlp(input_dim: int, output_dim: int, hidden_layers: Iterable[int], activation: str, dropout: float) -> Any:
    from torch import nn

    layers: list[Any] = []
    previous = input_dim
    for width in hidden_layers:
        layers.extend([nn.Linear(previous, int(width)), activation_layer(activation)])
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def build_checkpoint_ratio_estimator(checkpoint: Mapping[str, Any]) -> Any:
    import torch
    from torch import nn

    class CheckpointRatioEstimator(nn.Module):
        def __init__(self, checkpoint_data: Mapping[str, Any]):
            super().__init__()
            scalers = checkpoint_data["scalers"]
            training_config = checkpoint_data["training_config"]

            feature_mean = np.asarray(scalers["feature"]["mean"], dtype=np.float32)
            feature_scale = np.asarray(scalers["feature"]["scale"], dtype=np.float32)
            theta_mean = np.asarray(scalers["theta"]["mean"], dtype=np.float32)
            theta_scale = np.asarray(scalers["theta"]["scale"], dtype=np.float32)
            log_r_mean = np.asarray(scalers["log_r"]["mean"], dtype=np.float32)
            log_r_scale = np.asarray(scalers["log_r"]["scale"], dtype=np.float32)

            self.n_features = int(feature_mean.shape[1])
            self.network = build_mlp(
                self.n_features + int(theta_mean.shape[1]),
                1,
                training_config["hidden_layers"],
                str(training_config.get("activation", "tanh")),
                float(training_config.get("dropout", 0.0)),
            )
            state_dict = checkpoint_data["state_dict"]
            if all(key.startswith("network.") for key in state_dict):
                state_dict = {key.removeprefix("network."): value for key, value in state_dict.items()}
            self.network.load_state_dict(state_dict)

            self.register_buffer("feature_mean", torch.as_tensor(feature_mean))
            self.register_buffer("feature_scale", torch.as_tensor(feature_scale))
            self.register_buffer("theta_mean", torch.as_tensor(theta_mean))
            self.register_buffer("theta_scale", torch.as_tensor(theta_scale))
            self.register_buffer("log_r_mean", torch.as_tensor(log_r_mean))
            self.register_buffer("log_r_scale", torch.as_tensor(log_r_scale))

        def forward(self, inputs: Any) -> Any:
            features = inputs[:, : self.n_features]
            theta = inputs[:, self.n_features :]
            features_scaled = (features - self.feature_mean) / self.feature_scale
            theta_scaled = (theta - self.theta_mean) / self.theta_scale
            scaled_log_r = self.network(torch.cat([features_scaled, theta_scaled], dim=1))
            return scaled_log_r * self.log_r_scale + self.log_r_mean

    return CheckpointRatioEstimator(checkpoint)


def load_model(model_config: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, str]:
    import torch

    if model_config.get("kind", "checkpoint") != "checkpoint":
        raise ValueError("Only checkpoint models are supported.")

    requested_device = str(model_config.get("device", "auto"))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    checkpoint = torch.load(resolve_path(config, model_config["path"]), map_location=device, weights_only=False)
    model = build_checkpoint_ratio_estimator(checkpoint).to(device)
    model.eval()
    return model, device


def model_output_name(model_config: Mapping[str, Any]) -> str:
    name = str(model_config.get("name") or Path(model_config["path"]).stem)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def model_display_name(model_config: Mapping[str, Any]) -> str:
    return str(model_config.get("method") or model_config.get("name") or Path(model_config["path"]).stem).upper()


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return sigma


def calibration_output_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"{model_name}_parameter_calibration.json"


def calibration_sigma_plot_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"{model_name}_parameter_sigma.png"


def score_calibration_output_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"{model_name}_score_calibration.json"


def build_parameter_calibration(
    model_config: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
) -> ParameterCalibration | None:
    calibration_config = config.get("calibration", {})
    if not bool(calibration_config.get("enabled", True)):
        return None

    table_path = resolve_path(
        config,
        model_config.get("calibration_table", calibration_config.get("prediction_table", DEFAULT_PREDICTION_TABLE_PATH)),
    )
    if not table_path.exists():
        print(f"  calibration skipped: prediction table not found at {table_path}", flush=True)
        return None

    frame = pd.read_csv(table_path, usecols=lambda column: column in {"method", "log_r_true", "log_r_pred"})
    method_name = model_display_name(model_config)
    if "method" in frame.columns:
        method_frame = frame[frame["method"].astype(str).str.upper() == method_name].copy()
        if method_frame.empty:
            method_frame = frame[frame["method"].astype(str).str.upper() == Path(model_config["path"]).stem.upper()].copy()
    else:
        method_frame = frame.copy()
    if method_frame.empty:
        print(f"  calibration skipped: no rows for {method_name} in {table_path}", flush=True)
        return None

    values = method_frame[["log_r_true", "log_r_pred"]].replace([np.inf, -np.inf], np.nan)
    residual = values["log_r_pred"].to_numpy(dtype=float) - values["log_r_true"].to_numpy(dtype=float)
    min_total = int(calibration_config.get("min_total_count", 100))
    finite_residual = np.isfinite(residual)
    if int(finite_residual.sum()) < min_total:
        print(f"  calibration skipped: only {int(finite_residual.sum())} residual rows for {method_name}", flush=True)
        return None

    theta_values, theta_columns, theta_source = load_calibration_thetas(method_frame, config, calibration_config)
    residual = residual[finite_residual]
    theta_values = theta_values[finite_residual]

    c1_bins = int(calibration_config.get("theta_c1_bins", calibration_config.get("theta_bins", 12)))
    c2_bins = int(calibration_config.get("theta_c2_bins", calibration_config.get("theta_bins", 12)))
    min_count = int(calibration_config.get("theta_min_bin_count", calibration_config.get("min_bin_count", 80)))
    sigma_floor = float(calibration_config.get("sigma_floor", 0.02))
    apply_bias_correction = bool(calibration_config.get("apply_bias_correction", False))
    c1_edges = np.linspace(float(np.nanmin(theta_values[:, 0])), float(np.nanmax(theta_values[:, 0])), c1_bins + 1)
    c2_edges = np.linspace(float(np.nanmin(theta_values[:, 1])), float(np.nanmax(theta_values[:, 1])), c2_bins + 1)
    global_bias = float(np.median(residual))
    global_sigma = max(robust_sigma(residual), sigma_floor)
    bias = np.full((c1_bins, c2_bins), global_bias, dtype=float)
    sigma = np.full((c1_bins, c2_bins), global_sigma, dtype=float)
    counts = np.zeros((c1_bins, c2_bins), dtype=int)

    c1_indices = np.clip(np.searchsorted(c1_edges, theta_values[:, 0], side="right") - 1, 0, c1_bins - 1)
    c2_indices = np.clip(np.searchsorted(c2_edges, theta_values[:, 1], side="right") - 1, 0, c2_bins - 1)
    for c1_index in range(c1_bins):
        for c2_index in range(c2_bins):
            mask = (c1_indices == c1_index) & (c2_indices == c2_index)
            counts[c1_index, c2_index] = int(mask.sum())
            if counts[c1_index, c2_index] >= min_count:
                bias[c1_index, c2_index] = float(np.median(residual[mask]))
                sigma[c1_index, c2_index] = max(robust_sigma(residual[mask]), sigma_floor)

    output = {
        "model": model_output_name(model_config),
        "method": method_name,
        "source": str(table_path),
        "theta_source": theta_source,
        "theta_columns": theta_columns,
        "theta_c1_edges": c1_edges.tolist(),
        "theta_c2_edges": c2_edges.tolist(),
        "bias": bias.tolist(),
        "sigma": sigma.tolist(),
        "counts": counts.tolist(),
        "sigma_floor": sigma_floor,
        "global_bias": global_bias,
        "global_sigma": global_sigma,
        "residual_definition": "log_r_pred - log_r_true",
        "calibration_type": "parameter_binned",
        "apply_bias_correction": apply_bias_correction,
        "sparse_bin_fallback": "Sparse theta bins fall back to the global residual calibration.",
    }
    output_path = calibration_output_path(output_dir, model_output_name(model_config))
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  calibration: wrote {output_path}", flush=True)
    return ParameterCalibration(
        c1_edges,
        c2_edges,
        bias,
        sigma,
        counts,
        sigma_floor,
        global_bias,
        global_sigma,
        apply_bias_correction,
    )


def build_score_calibration(
    model_config: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
) -> ScoreCalibration:
    """Estimate score centering and covariance from held-out numerator events."""

    score_config = config.get("score_constraints", {})
    table_path = resolve_path(
        config,
        model_config.get(
            "score_calibration_table",
            score_config.get("prediction_table", DEFAULT_PREDICTION_TABLE_PATH),
        ),
    )
    names = coefficient_names(config)[:2]
    theta_columns = [f"theta0_{name}" for name in names]
    score_columns = [f"score_pred_{name}" for name in names]
    required = ["method", "y", *theta_columns, *score_columns]
    frame = pd.read_csv(table_path, usecols=required)
    method_name = model_display_name(model_config)
    frame = frame[frame["method"].astype(str).str.upper() == method_name]
    frame = frame[frame["y"].astype(float) == 1.0].replace([np.inf, -np.inf], np.nan).dropna()
    min_count = int(score_config.get("min_count_per_theta", 50))

    theta_values: list[np.ndarray] = []
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    counts: list[int] = []
    for theta_value, group in frame.groupby(theta_columns, sort=False):
        if len(group) < min_count:
            continue
        scores = group[score_columns].to_numpy(dtype=float)
        covariance = np.cov(scores, rowvar=False, ddof=1)
        if covariance.shape != (len(score_columns), len(score_columns)) or not np.isfinite(covariance).all():
            continue
        theta_values.append(np.asarray(theta_value, dtype=float))
        means.append(scores.mean(axis=0))
        covariances.append(covariance)
        counts.append(len(scores))

    if not theta_values:
        raise RuntimeError(
            f"No score-calibration groups with at least {min_count} numerator events for {method_name} in {table_path}."
        )

    theta_array = np.stack(theta_values)
    coordinate_scale = np.ptp(theta_array, axis=0)
    coordinate_scale = np.where(coordinate_scale > 0.0, coordinate_scale, 1.0)
    calibration = ScoreCalibration(
        theta=theta_array,
        mean=np.stack(means),
        covariance=np.stack(covariances),
        counts=np.asarray(counts, dtype=int),
        coordinate_scale=coordinate_scale,
        neighbors=int(score_config.get("interpolation_neighbors", 4)),
        distance_power=float(score_config.get("interpolation_distance_power", 2.0)),
        covariance_ridge_fraction=float(score_config.get("covariance_ridge_fraction", 1.0e-4)),
        include_interpolation_uncertainty=bool(score_config.get("include_interpolation_uncertainty", True)),
    )
    output = {
        "model": model_output_name(model_config),
        "method": method_name,
        "source": str(table_path),
        "selection": "held-out numerator rows (y=1)",
        "theta_columns": theta_columns,
        "score_columns": score_columns,
        "theta": calibration.theta.tolist(),
        "mean": calibration.mean.tolist(),
        "covariance": calibration.covariance.tolist(),
        "counts": calibration.counts.tolist(),
        "coordinate_scale": calibration.coordinate_scale.tolist(),
        "interpolation_neighbors": calibration.neighbors,
        "interpolation_distance_power": calibration.distance_power,
        "covariance_ridge_fraction": calibration.covariance_ridge_fraction,
        "include_interpolation_uncertainty": calibration.include_interpolation_uncertainty,
        "statistic": "delta_score^T [N*event_covariance + N^2*mean_covariance]^-1 delta_score",
    }
    output_path = score_calibration_output_path(output_dir, model_output_name(model_config))
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  score calibration: wrote {output_path} ({len(theta_array)} theta points)", flush=True)
    return calibration


def load_calibration_thetas(
    method_frame: pd.DataFrame,
    config: Mapping[str, Any],
    calibration_config: Mapping[str, Any],
) -> tuple[np.ndarray, list[str], str]:
    names = coefficient_names(config)
    theta_columns = [f"theta0_{names[0]}", f"theta0_{names[1]}"]
    metadata_path = resolve_path(config, calibration_config.get("ratio_metadata_table", "table_outputs/madminer_style_training/ratio_test.csv"))
    metadata = pd.read_csv(metadata_path, usecols=theta_columns)
    if len(metadata) != len(method_frame):
        raise ValueError(f"{len(metadata)} theta rows for {len(method_frame)} prediction rows")
    return metadata.to_numpy(dtype=float), theta_columns, str(metadata_path)


def plot_parameter_calibration(calibration: ParameterCalibration, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.0, 5.6))
    image = axis.pcolormesh(calibration.c1_edges, calibration.c2_edges, calibration.sigma.T, shading="auto")
    figure.colorbar(image, ax=axis, label="sigma(log r residual)")
    axis.set_xlabel(r"$f_W v^2 / \Lambda^2$")
    axis.set_ylabel(r"$f_{WW} v^2 / \Lambda^2$")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def theta_grid(config: Mapping[str, Any], c1_bins: int | None, c2_bins: int | None) -> tuple[np.ndarray, np.ndarray, list[ThetaPoint]]:
    grid = config["theta_grid"]
    c1 = np.linspace(float(grid["c1_min"]), float(grid["c1_max"]), int(c1_bins or grid.get("c1_bins", 50)))
    c2 = np.linspace(float(grid["c2_min"]), float(grid["c2_max"]), int(c2_bins or grid.get("c2_bins", 50)))
    return c1, c2, [ThetaPoint(float(x), float(y)) for x in c1 for y in c2]


def local_theta_grid(
    config: Mapping[str, Any],
    bounds: tuple[float, float, float, float],
    local_config: Mapping[str, Any],
    theta_true: ThetaPoint | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ThetaPoint]]:
    c1_min, c1_max, c2_min, c2_max = bounds
    bins = int(local_config.get("bins", 80))
    c1 = np.linspace(c1_min, c1_max, bins)
    c2 = np.linspace(c2_min, c2_max, bins)

    if theta_true is not None:
        # Closure diagnostics must evaluate the known truth exactly.  With an
        # even number of bins a symmetric linspace otherwise misses its centre.
        if c1_min <= theta_true.c1 <= c1_max:
            c1[np.argmin(np.abs(c1 - theta_true.c1))] = theta_true.c1
            c1.sort()
        if c2_min <= theta_true.c2 <= c2_max:
            c2[np.argmin(np.abs(c2 - theta_true.c2))] = theta_true.c2
            c2.sort()

    return c1, c2, [ThetaPoint(float(x), float(y)) for x in c1 for y in c2]


def local_scan_bounds(
    coarse_frame: pd.DataFrame,
    config: Mapping[str, Any],
    heatmap_config: Mapping[str, Any],
    local_config: Mapping[str, Any],
    theta_true: ThetaPoint,
) -> tuple[float, float, float, float]:
    grid = config["theta_grid"]
    full_c1_min = float(grid["c1_min"])
    full_c1_max = float(grid["c1_max"])
    full_c2_min = float(grid["c2_min"])
    full_c2_max = float(grid["c2_max"])
    levels = [float(level) for level in heatmap_config.get("cl_levels", [2.278868566, 5.991464547, 11.61828598])]
    level = float(local_config.get("range_q_level", max(levels)))
    range_column = str(local_config.get("range_column", "q_relative"))
    region = coarse_frame[coarse_frame[range_column] <= level]
    if region.empty:
        # No coarse points inside the outermost CL: fall back to the single best
        # coarse point so the local scan stays tightly focused around the
        # calibrated minimum.
        best = coarse_frame.loc[coarse_frame[range_column].idxmin()]
        region = pd.DataFrame([best])

    c1_min = float(region["c1"].min())
    c1_max = float(region["c1"].max())
    c2_min = float(region["c2"].min())
    c2_max = float(region["c2"].max())
    coarse_c1 = np.sort(coarse_frame["c1"].unique())
    coarse_c2 = np.sort(coarse_frame["c2"].unique())
    c1_step = float(np.median(np.diff(coarse_c1))) if len(coarse_c1) > 1 else 1.0
    c2_step = float(np.median(np.diff(coarse_c2))) if len(coarse_c2) > 1 else 1.0
    margin_fraction = float(local_config.get("range_margin_fraction", 0.25))
    c1_margin = max(c1_step, margin_fraction * max(c1_max - c1_min, c1_step))
    c2_margin = max(c2_step, margin_fraction * max(c2_max - c2_min, c2_step))
    return (
        max(full_c1_min, c1_min - c1_margin),
        min(full_c1_max, c1_max + c1_margin),
        max(full_c2_min, c2_min - c2_margin),
        min(full_c2_max, c2_max + c2_margin),
    )


def truth_centered_scan_bounds(
    coarse_frame: pd.DataFrame,
    config: Mapping[str, Any],
    local_config: Mapping[str, Any],
    theta_true: ThetaPoint,
) -> tuple[float, float, float, float]:
    """Return a fine closure-test window guaranteed to contain theta_true."""

    grid = config["theta_grid"]
    coarse_c1 = np.sort(coarse_frame["c1"].unique())
    coarse_c2 = np.sort(coarse_frame["c2"].unique())
    c1_step = float(np.median(np.diff(coarse_c1))) if len(coarse_c1) > 1 else 1.0
    c2_step = float(np.median(np.diff(coarse_c2))) if len(coarse_c2) > 1 else 1.0
    steps = float(local_config.get("truth_half_width_coarse_steps", 1.5))
    c1_half_width = float(local_config.get("truth_half_width_c1", steps * c1_step))
    c2_half_width = float(local_config.get("truth_half_width_c2", steps * c2_step))
    return (
        max(float(grid["c1_min"]), theta_true.c1 - c1_half_width),
        min(float(grid["c1_max"]), theta_true.c1 + c1_half_width),
        max(float(grid["c2_min"]), theta_true.c2 - c2_half_width),
        min(float(grid["c2_max"]), theta_true.c2 + c2_half_width),
    )


def evaluate_grid(
    model: Any,
    device: str,
    events: np.ndarray,
    points: list[ThetaPoint],
    batch_size: int,
    scan_stage: str,
    theta_batch_size: int,
    calibration: ParameterCalibration | None,
    score_calibration: ScoreCalibration | None,
) -> pd.DataFrame:
    """Evaluate the summed log-ratio on every grid point.

    Features are moved to the target device once and reused for every theta
    point, avoiding the N_events × N_theta cross-product allocation that
    dominated runtime in the previous implementation.  Each theta point now
    gets a single forward pass of size N_events (chunked by batch_size for
    memory safety), rather than ~N_events/batch_size passes of interleaved data.
    """
    import torch

    rows: list[dict[str, Any]] = []
    event_count = len(events)

    # Move the feature array to the target device once; reuse for every theta.
    feature_tensor = torch.as_tensor(events.astype(np.float32), dtype=torch.float32, device=device)

    for chunk_start in range(0, len(points), theta_batch_size):
        chunk_points = points[chunk_start : chunk_start + theta_batch_size]
        chunk_predictions: list[np.ndarray] = []
        chunk_score_sums: list[np.ndarray] = []

        for point in chunk_points:
            predictions_for_point: list[torch.Tensor] = []
            scores_for_point: list[torch.Tensor] = []
            theta_value = torch.as_tensor(point.as_array(), dtype=torch.float32, device=device)
            for start in range(0, event_count, batch_size):
                features_batch = feature_tensor[start : start + batch_size]
                if score_calibration is None:
                    with torch.no_grad():
                        theta_batch = theta_value.unsqueeze(0).expand(len(features_batch), -1)
                        inputs = torch.cat([features_batch, theta_batch], dim=1)
                        prediction = model(inputs).squeeze(-1)
                else:
                    # Differentiate with respect to *physical* theta.  The
                    # checkpoint wrapper performs scaling internally, so this
                    # gradient is already the physical marginal score.
                    theta_batch = theta_value.unsqueeze(0).expand(len(features_batch), -1).clone().requires_grad_(True)
                    inputs = torch.cat([features_batch, theta_batch], dim=1)
                    prediction = model(inputs).squeeze(-1)
                    score = torch.autograd.grad(prediction.sum(), theta_batch, create_graph=False)[0]
                    scores_for_point.append(score.detach())
                predictions_for_point.append(prediction.detach())
            chunk_predictions.append(torch.cat(predictions_for_point).cpu().numpy())
            if scores_for_point:
                chunk_score_sums.append(torch.cat(scores_for_point).sum(dim=0).cpu().numpy())

        predictions = np.stack(chunk_predictions)  # (chunk_size, N_events)

        sums = predictions.sum(axis=1, dtype=np.float64)
        means = predictions.mean(axis=1)
        stds = predictions.std(axis=1, ddof=1) if event_count > 1 else np.zeros(len(chunk_points), dtype=float)
        corrected_sums, corrected_means, variance_sums, mean_sigmas = calibrated_summaries(predictions, chunk_points, calibration)
        if score_calibration is not None:
            score_sums = np.stack(chunk_score_sums).astype(np.float64)
            score_means, centered_score_sums, q_scores = calibrated_score_test(
                score_sums, chunk_points, event_count, score_calibration
            )
        else:
            score_sums = np.full((len(chunk_points), 2), np.nan)
            score_means = np.full((len(chunk_points), 2), np.nan)
            centered_score_sums = np.full((len(chunk_points), 2), np.nan)
            q_scores = np.full(len(chunk_points), np.nan)

        for index, values in enumerate(zip(chunk_points, sums, means, stds, corrected_sums, corrected_means, variance_sums, mean_sigmas)):
            point, sum_log_r, mean_log_r, std_log_r, corrected_sum, corrected_mean, variance_sum, mean_sigma = values
            rows.append(
                {
                    "c1": point.c1,
                    "c2": point.c2,
                    "mean_log_r": float(mean_log_r),
                    "sum_log_r": float(sum_log_r),
                    "minus2_sum_log_r": -2.0 * float(sum_log_r),
                    "mean_log_r_calibrated": float(corrected_mean),
                    "sum_log_r_calibrated": float(corrected_sum),
                    "minus2_sum_log_r_calibrated": -2.0 * float(corrected_sum),
                    "var_sum_log_r_nn": float(variance_sum),
                    "sigma_sum_log_r_nn": float(math.sqrt(max(variance_sum, 0.0))),
                    "mean_sigma_log_r_nn": float(mean_sigma),
                    "std_log_r": float(std_log_r),
                    "score_sum_c1": float(score_sums[index, 0]),
                    "score_sum_c2": float(score_sums[index, 1]),
                    "score_expected_mean_c1": float(score_means[index, 0]),
                    "score_expected_mean_c2": float(score_means[index, 1]),
                    "score_centered_sum_c1": float(centered_score_sums[index, 0]),
                    "score_centered_sum_c2": float(centered_score_sums[index, 1]),
                    "q_score": float(q_scores[index]),
                    "n_events": int(event_count),
                    "scan_stage": scan_stage,
                }
            )

        done = min(chunk_start + len(chunk_points), len(points))
        print(f"    evaluated {done:,}/{len(points):,} grid points", flush=True)

    return pd.DataFrame(rows)


def calibrated_summaries(
    predictions: np.ndarray,
    points: list[ThetaPoint],
    calibration: ParameterCalibration | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if calibration is None:
        sums = predictions.sum(axis=1, dtype=np.float64)
        return sums, predictions.mean(axis=1), np.zeros(len(points), dtype=float), np.zeros(len(points), dtype=float)

    bias, sigma = calibration.lookup_many(points)
    corrected = predictions - bias[:, None]
    # Promote to float64 before squaring to avoid precision loss for large N_events.
    sigma64 = sigma.astype(np.float64)
    n_events = np.float64(predictions.shape[1])
    return (
        corrected.sum(axis=1, dtype=np.float64),
        corrected.mean(axis=1),
        np.square(sigma64) * n_events,  # Var[sum] = N * sigma_per_event^2
        sigma64,
    )


def calibrated_score_test(
    score_sums: np.ndarray,
    points: Sequence[ThetaPoint],
    event_count: int,
    calibration: ScoreCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interpolated score means, centered sums, and score-test q."""

    score_means, score_covariances, mean_covariances = calibration.lookup_many(points)
    centered = np.asarray(score_sums, dtype=float) - float(event_count) * score_means
    total_covariances = (
        float(event_count) * score_covariances
        + float(event_count) ** 2 * mean_covariances
    )
    q_score = np.array(
        [
            max(float(vector @ np.linalg.pinv(covariance) @ vector), 0.0)
            for vector, covariance in zip(centered, total_covariances)
        ],
        dtype=float,
    )
    return score_means, centered, q_score


def add_q_columns(
    frame: pd.DataFrame,
    variance_floor: float,
    correlation_length: float | None,
    min_correlation: float,
) -> pd.DataFrame:
    frame = frame.copy()

    # Uncalibrated test statistic: kept for reference and backward compatibility.
    frame["q_relative"] = frame["minus2_sum_log_r"] - frame["minus2_sum_log_r"].min()

    # Calibrated best-fit point (maximum calibrated log-likelihood).
    best_index = frame["sum_log_r_calibrated"].idxmax()
    best_sum = float(frame.loc[best_index, "sum_log_r_calibrated"])
    best_var = float(frame.loc[best_index, "var_sum_log_r_nn"])
    best_c1 = float(frame.loc[best_index, "c1"])
    best_c2 = float(frame.loc[best_index, "c2"])

    # delta >= 0 by construction (best_sum is the maximum calibrated log-L).
    delta = np.maximum(best_sum - frame["sum_log_r_calibrated"].astype(float), 0.0)
    independent_variance = np.maximum(best_var + frame["var_sum_log_r_nn"].astype(float), variance_floor)
    theta_distance = np.sqrt(
        np.square(frame["c1"].astype(float) - best_c1)
        + np.square(frame["c2"].astype(float) - best_c2)
    )

    if correlation_length is not None and correlation_length > 0.0:
        rho = np.maximum(
            np.exp(-0.5 * np.square(theta_distance / float(correlation_length))),
            float(min_correlation),
        )
        covariance = rho * np.sqrt(np.maximum(best_var * frame["var_sum_log_r_nn"].astype(float), 0.0))
        variance = np.maximum(independent_variance - 2.0 * covariance, variance_floor)
    else:
        rho = np.zeros(len(frame), dtype=float)
        variance = independent_variance

    frame["theta_distance_from_calibrated_best"] = theta_distance
    frame["nn_error_correlation"] = rho
    frame["delta_sum_log_r_calibrated"] = delta
    frame["sigma_delta_log_r_nn_independent"] = np.sqrt(independent_variance)
    frame["sigma_delta_log_r_nn"] = np.sqrt(variance)

    # q_calibrated: proper Wilks profile-likelihood-ratio test statistic.
    # q(θ) = 2*(ΣlogL_cal(θ̂) − ΣlogL_cal(θ))  ≥ 0 by construction.
    # Under Wilks' theorem and an unbiased estimator, follows chi-sq(n_params).
    # The chi-sq CL thresholds in cl_levels apply directly to this column.
    frame["q_calibrated"] = 2.0 * delta

    # q_error_aware: conservative minimum of the proper Wilks statistic and
    # the signal-to-noise ratio squared (delta² / independent_variance).
    # Using *independent* variance (not correlation-corrected) prevents the
    # statistic from collapsing to zero when the best and test points are
    # close together (the zeroing-out failure mode of the previous formula).
    # Interpretation: q_error_aware ≤ q_calibrated always; it is reduced when
    # NN uncertainty dominates the log-L difference. chi-sq(2) thresholds are
    # approximate (conservative) for this combined statistic.
    frame["q_error_aware_independent"] = np.square(delta) / np.maximum(independent_variance, variance_floor)
    frame["q_error_aware"] = np.minimum(frame["q_calibrated"], frame["q_error_aware_independent"])

    return frame


def grid_to_matrix(frame: pd.DataFrame, c1_values: np.ndarray, c2_values: np.ndarray, column: str) -> np.ndarray:
    pivot = frame.pivot(index="c2", columns="c1", values=column).reindex(index=c2_values, columns=c1_values)
    return pivot.to_numpy(dtype=float)


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    return np.concatenate([[values[0] - (mids[0] - values[0])], mids, [values[-1] + (values[-1] - mids[-1])]])


def plot_heatmap(
    frame: pd.DataFrame,
    c1_values: np.ndarray,
    c2_values: np.ndarray,
    theta_true: ThetaPoint,
    output_path: Path,
    title: str,
    labels: tuple[str, str],
    heatmap_config: Mapping[str, Any],
    column: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    values = grid_to_matrix(frame, c1_values, c2_values, column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.8))
    image = plt.pcolormesh(
        centers_to_edges(c1_values),
        centers_to_edges(c2_values),
        values,
        shading="auto",
        vmin=0.0,
        vmax=float(heatmap_config.get("q_color_vmax", max(heatmap_config.get("cl_levels", [11.61828598])) * 1.25)),
    )
    plt.colorbar(image, label=column)

    handles = draw_contours(values, c1_values, c2_values, heatmap_config)
    best = frame.loc[frame[column].idxmin()]
    if bool(heatmap_config.get("show_true_point", True)):
        plt.scatter([theta_true.c1], [theta_true.c2], c="white", edgecolors="black", marker="*", s=110)
        handles.append(
            Line2D([0], [0], color="black", marker="*", linestyle="None", markersize=10, markerfacecolor="white", label="theta_true")
        )
    plt.scatter([best["c1"]], [best["c2"]], c="none", edgecolors="red", marker="o", s=95)
    handles.append(Line2D([0], [0], color="red", marker="o", linestyle="None", markersize=8, markerfacecolor="none", label="min q"))

    if bool(heatmap_config.get("zoom_to_contours", True)):
        zoom_to_q_region(frame, c1_values, c2_values, theta_true, heatmap_config, column)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.legend(handles=handles, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def draw_contours(values: np.ndarray, c1_values: np.ndarray, c2_values: np.ndarray, heatmap_config: Mapping[str, Any]) -> list[Any]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    levels = [float(level) for level in heatmap_config.get("cl_levels", [2.278868566, 5.991464547, 11.61828598])]
    labels = list(heatmap_config.get("cl_labels", ["68% CL", "95% CL", "99.7% CL"]))
    level_pairs = [(level, labels[index] if index < len(labels) else f"q={level:.3g}") for index, level in enumerate(levels)]
    level_pairs = [(level, label) for level, label in level_pairs if np.nanmin(values) <= level <= np.nanmax(values)]
    if not level_pairs:
        return []

    colors = ["#ffffff", "#ff7f0e", "#e6007e", "#00e5ff", "#ffd700"]
    c1_mesh, c2_mesh = np.meshgrid(c1_values, c2_values)
    level_values = [level for level, _ in level_pairs]
    plt.contour(c1_mesh, c2_mesh, values, levels=level_values, colors=["#111111"] * len(level_values), linewidths=4.0)
    contour = plt.contour(c1_mesh, c2_mesh, values, levels=level_values, colors=colors[: len(level_pairs)], linewidths=2.4)
    label_by_level = {level: label for level, label in level_pairs}
    texts = plt.clabel(contour, contour.levels, inline=True, fmt=label_by_level, fontsize=9)
    for text in texts:
        text.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})
    return [
        Line2D([0], [0], color=colors[index], lw=2.6, label=label_by_level[level])
        for index, level in enumerate(contour.levels)
    ]


def zoom_to_q_region(
    frame: pd.DataFrame,
    c1_values: np.ndarray,
    c2_values: np.ndarray,
    theta_true: ThetaPoint,
    heatmap_config: Mapping[str, Any],
    column: str,
) -> None:
    import matplotlib.pyplot as plt

    levels = [float(level) for level in heatmap_config.get("cl_levels", [2.278868566, 5.991464547, 11.61828598])]
    best = frame.loc[frame[column].idxmin()]
    kept = frame[frame[column] <= max(levels)]
    x_points = [float(best["c1"]), theta_true.c1] + kept["c1"].astype(float).tolist()
    y_points = [float(best["c2"]), theta_true.c2] + kept["c2"].astype(float).tolist()
    x_min, x_max = min(x_points), max(x_points)
    y_min, y_max = min(y_points), max(y_points)
    full_x_width = float(c1_values.max() - c1_values.min())
    full_y_width = float(c2_values.max() - c2_values.min())
    x_margin = max(0.12 * max(x_max - x_min, 1.0e-9), 0.02 * full_x_width)
    y_margin = max(0.12 * max(y_max - y_min, 1.0e-9), 0.02 * full_y_width)
    plt.xlim(max(float(c1_values.min()), x_min - x_margin), min(float(c1_values.max()), x_max + x_margin))
    plt.ylim(max(float(c2_values.min()), y_min - y_margin), min(float(c2_values.max()), y_max + y_margin))


def selected_entries(entries: list[dict[str, Any]], requested: set[str] | None, name_fn) -> list[dict[str, Any]]:
    if not requested:
        return entries
    selected = [entry for entry in entries if name_fn(entry).lower() in requested]
    missing = requested - {name_fn(entry).lower() for entry in selected}
    if missing:
        raise ValueError(f"Requested entries not found: {sorted(missing)}")
    return selected


def nearest_theta_row(frame: pd.DataFrame, theta: ThetaPoint) -> pd.Series:
    distances = np.square(frame["c1"].astype(float) - theta.c1) + np.square(frame["c2"].astype(float) - theta.c2)
    return frame.loc[distances.idxmin()]


def q_diagnostics(model_name: str, dataset_name: str, theta_true: ThetaPoint, frame: pd.DataFrame, q_frame: pd.DataFrame, cl_levels: Sequence[float]) -> dict[str, Any]:
    raw_best = frame.loc[frame["minus2_sum_log_r"].idxmin()]
    calibrated_best = frame.loc[frame["sum_log_r_calibrated"].idxmax()]
    error_aware_best = q_frame.loc[q_frame["q_error_aware"].idxmin()]
    score_best = q_frame.loc[q_frame["q_score"].idxmin()]
    truth = nearest_theta_row(q_frame, theta_true)
    diagnostics: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "theta_true_c1": theta_true.c1,
        "theta_true_c2": theta_true.c2,
        "best_raw_c1": float(raw_best["c1"]),
        "best_raw_c2": float(raw_best["c2"]),
        "best_calibrated_log_l_c1": float(calibrated_best["c1"]),
        "best_calibrated_log_l_c2": float(calibrated_best["c2"]),
        "best_error_aware_c1": float(error_aware_best["c1"]),
        "best_error_aware_c2": float(error_aware_best["c2"]),
        "best_score_c1": float(score_best["c1"]),
        "best_score_c2": float(score_best["c2"]),
        "truth_nearest_c1": float(truth["c1"]),
        "truth_nearest_c2": float(truth["c2"]),
        "truth_q_relative": float(truth.get("q_relative", np.nan)),
        "truth_q_calibrated": float(truth.get("q_calibrated", np.nan)),
        "truth_q_error_aware": float(truth.get("q_error_aware", np.nan)),
        "truth_q_score": float(truth.get("q_score", np.nan)),
        "truth_delta_sum_log_r_calibrated": float(truth.get("delta_sum_log_r_calibrated", np.nan)),
        "truth_sigma_delta_log_r_nn": float(truth.get("sigma_delta_log_r_nn", np.nan)),
        "n_events": int(truth.get("n_events", len(q_frame))),
    }
    diagnostics["truth_delta_sum_log_r_calibrated_per_event"] = diagnostics["truth_delta_sum_log_r_calibrated"] / max(diagnostics["n_events"], 1)
    for level in cl_levels:
        diagnostics[f"truth_inside_q_calibrated_{float(level):.6g}"] = bool(diagnostics["truth_q_calibrated"] <= float(level))
        diagnostics[f"truth_inside_q_error_aware_{float(level):.6g}"] = bool(diagnostics["truth_q_error_aware"] <= float(level))
        diagnostics[f"truth_inside_q_score_{float(level):.6g}"] = bool(diagnostics["truth_q_score"] <= float(level))
    return diagnostics


def run_constraints(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = load_config(args.config)
    config["coefficient_names"] = coefficient_names(config)
    output_dir = resolve_path(config, args.output_dir or config.get("output_dir", "table_outputs/constraints")).resolve()
    plot_output_dir = resolve_path(config, args.plot_output_dir or config.get("plot_output_dir", "plotting_outputs/constraints")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_config = config.get("heatmap", {})
    calibration_config = config.get("calibration", {})
    max_events = args.max_events if args.max_events is not None else heatmap_config.get("max_events", 5000)
    max_events = None if max_events is None else int(max_events)
    batch_size = int(args.batch_size or heatmap_config.get("batch_size", 4096))
    theta_batch_size = int(args.theta_batch_size or heatmap_config.get("theta_batch_size", 16))
    c1_values, c2_values, coarse_points = theta_grid(config, args.c1_bins or heatmap_config.get("c1_bins"), args.c2_bins or heatmap_config.get("c2_bins"))
    q_options = {
        "variance_floor": float(calibration_config.get("variance_floor", 1.0e-12)),
        "correlation_length": None if calibration_config.get("correlation_length", 1.0) is None else float(calibration_config.get("correlation_length", 1.0)),
        "min_correlation": float(calibration_config.get("min_correlation", 0.0)),
    }

    datasets = selected_entries(
        validation_datasets(config),
        {item.lower() for item in args.datasets} if args.datasets else None,
        lambda entry: str(entry.get("name") or entry.get("theta_tag")),
    )
    models = selected_entries(
        list(config["models"]),
        {item.lower() for item in args.models} if args.models else None,
        model_output_name,
    )

    results: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    calibrations: dict[str, ParameterCalibration] = {}
    score_calibrations: dict[str, ScoreCalibration] = {}
    # Cache loaded model weights so each model is read from disk only once,
    # not once per dataset × model combination.
    model_cache: dict[str, tuple[Any, str]] = {}
    labels = tuple(config["coefficient_names"][:2])

    for dataset_index, dataset in enumerate(datasets):
        dataset_name = str(dataset.get("name") or dataset.get("theta_tag") or f"dataset_{dataset_index:03d}")
        theta_true = ThetaPoint.from_value(dataset["theta_true"])
        print(f"Loading {dataset_name} from {resolve_path(config, dataset['event_file'])} (max_events={max_events})", flush=True)
        events = load_feature_array(dataset["event_file"], config, max_events)
        print(f"  using {len(events):,} events", flush=True)

        for model_config in models:
            model_name = model_output_name(model_config)
            calibration = calibrations.get(model_name)
            if calibration is None and bool(calibration_config.get("enabled", False)):
                calibration = build_parameter_calibration(model_config, config, output_dir)
                if calibration is not None:
                    calibrations[model_name] = calibration
                    plot_parameter_calibration(calibration, calibration_sigma_plot_path(plot_output_dir, model_name), f"{model_name.upper()} parameter-binned residual sigma")
            score_calibration = score_calibrations.get(model_name)
            if score_calibration is None:
                score_calibration = build_score_calibration(model_config, config, output_dir)
                score_calibrations[model_name] = score_calibration

            print(f"  evaluating {model_name} on {len(coarse_points):,} coarse grid points", flush=True)
            if model_name not in model_cache:
                print(f"  loading model weights: {model_name}", flush=True)
                model_cache[model_name] = load_model(model_config, config)
            model, device = model_cache[model_name]
            coarse_frame = add_q_columns(
                evaluate_grid(model, device, events, coarse_points, batch_size, "coarse", theta_batch_size, calibration, score_calibration),
                **q_options,
            )
            (
                frame,
                score_frame,
                score_c1_values,
                score_c2_values,
                likelihood_frame,
                likelihood_c1_values,
                likelihood_c2_values,
            ) = maybe_refine_locally(
                model,
                device,
                events,
                coarse_frame,
                theta_true,
                config,
                heatmap_config,
                batch_size,
                theta_batch_size,
                calibration,
                score_calibration,
                q_options,
            )
            result = write_outputs(
                output_dir,
                plot_output_dir,
                model_name,
                dataset_name,
                theta_true,
                frame,
                coarse_frame,
                c1_values,
                c2_values,
                score_frame,
                score_c1_values,
                score_c2_values,
                likelihood_frame,
                likelihood_c1_values,
                likelihood_c2_values,
                labels,
                heatmap_config,
            )
            diagnostics = q_diagnostics(model_name, dataset_name, theta_true, frame, score_frame, heatmap_config.get("cl_levels", [2.278868566, 5.991464547, 11.61828598]))
            diagnostic_rows.append(diagnostics)
            results.append({**result, **manifest_summary(theta_true, frame, score_frame, diagnostics, len(events), model_name, dataset_name, output_dir, plot_output_dir)})

    (output_dir / "likelihood_heatmap_manifest.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(output_dir / "constraint_truth_diagnostics.csv", index=False)
    return results


def maybe_refine_locally(
    model: Any,
    device: str,
    events: np.ndarray,
    coarse_frame: pd.DataFrame,
    theta_true: ThetaPoint,
    config: Mapping[str, Any],
    heatmap_config: Mapping[str, Any],
    batch_size: int,
    theta_batch_size: int,
    calibration: ParameterCalibration,
    score_calibration: ScoreCalibration,
    q_options: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    local_config = heatmap_config.get("local_scan", {})
    if not bool(local_config.get("enabled", True)):
        c1_values, c2_values, _ = theta_grid(config, heatmap_config.get("c1_bins"), heatmap_config.get("c2_bins"))
        return coarse_frame, coarse_frame, c1_values, c2_values, coarse_frame, c1_values, c2_values

    # Score closure: use a fine window around the known validation truth.  A
    # separate full coarse score map remains in the outputs to expose any
    # disconnected or boundary minima rather than hiding them.
    score_bounds = truth_centered_scan_bounds(coarse_frame, config, local_config, theta_true)
    score_c1_values, score_c2_values, score_points = local_theta_grid(config, score_bounds, local_config, theta_true)
    c1_min, c1_max, c2_min, c2_max = score_bounds
    print(
        f"  refining score-closure range c1=[{c1_min:.4g}, {c1_max:.4g}], "
        f"c2=[{c2_min:.4g}, {c2_max:.4g}] on {len(score_points):,} local points",
        flush=True,
    )
    score_local = add_q_columns(
        evaluate_grid(model, device, events, score_points, batch_size, "score_local", theta_batch_size, calibration, score_calibration),
        **q_options,
    )

    # Raw likelihood diagnostic: refine independently around the raw
    # likelihood region and skip score gradients for this second pass.
    likelihood_config = {**local_config, "range_column": "q_relative"}
    likelihood_bounds = local_scan_bounds(coarse_frame, config, heatmap_config, likelihood_config, theta_true)
    likelihood_c1_values, likelihood_c2_values, likelihood_points = local_theta_grid(
        config, likelihood_bounds, local_config, theta_true
    )
    c1_min, c1_max, c2_min, c2_max = likelihood_bounds
    print(
        f"  refining raw-likelihood range c1=[{c1_min:.4g}, {c1_max:.4g}], "
        f"c2=[{c2_min:.4g}, {c2_max:.4g}] on {len(likelihood_points):,} local points",
        flush=True,
    )
    likelihood_local = add_q_columns(
        evaluate_grid(model, device, events, likelihood_points, batch_size, "likelihood_local", theta_batch_size, calibration, None),
        **q_options,
    )
    frame = add_q_columns(
        pd.concat(
            [
                coarse_frame.drop(columns=q_column_names(), errors="ignore"),
                score_local.drop(columns=q_column_names(), errors="ignore"),
                likelihood_local.drop(columns=q_column_names(), errors="ignore"),
            ],
            ignore_index=True,
        ),
        **q_options,
    )
    return (
        frame,
        score_local,
        score_c1_values,
        score_c2_values,
        likelihood_local,
        likelihood_c1_values,
        likelihood_c2_values,
    )


def q_column_names() -> list[str]:
    return [
        "q_relative",
        "q_calibrated",
        "q_error_aware",
        "q_error_aware_independent",
        "theta_distance_from_calibrated_best",
        "nn_error_correlation",
        "delta_sum_log_r_calibrated",
        "sigma_delta_log_r_nn_independent",
        "sigma_delta_log_r_nn",
    ]


def write_outputs(
    output_dir: Path,
    plot_output_dir: Path,
    model_name: str,
    dataset_name: str,
    theta_true: ThetaPoint,
    frame: pd.DataFrame,
    coarse_frame: pd.DataFrame,
    coarse_c1_values: np.ndarray,
    coarse_c2_values: np.ndarray,
    score_frame: pd.DataFrame,
    score_c1_values: np.ndarray,
    score_c2_values: np.ndarray,
    likelihood_frame: pd.DataFrame,
    likelihood_c1_values: np.ndarray,
    likelihood_c2_values: np.ndarray,
    labels: tuple[str, str],
    heatmap_config: Mapping[str, Any],
) -> dict[str, str]:
    prefix = f"{model_name}_{dataset_name}"
    table_path = output_dir / f"{prefix}_likelihood_grid.csv"
    score_table_path = output_dir / f"{prefix}_local_score_grid.csv"
    likelihood_table_path = output_dir / f"{prefix}_local_likelihood_grid.csv"
    score_heatmap_path = plot_output_dir / f"{prefix}_q_score_heatmap.png"
    global_score_heatmap_path = plot_output_dir / f"{prefix}_q_score_global_coarse_heatmap.png"
    likelihood_heatmap_path = plot_output_dir / f"{prefix}_q_likelihood_raw_heatmap.png"
    frame.to_csv(table_path, index=False)
    score_frame.to_csv(score_table_path, index=False)
    likelihood_frame.to_csv(likelihood_table_path, index=False)
    plot_heatmap(
        score_frame,
        score_c1_values,
        score_c2_values,
        theta_true,
        score_heatmap_path,
        f"{model_name.upper()} calibrated score test: {dataset_name}",
        labels,
        heatmap_config,
        "q_score",
    )
    plot_heatmap(
        coarse_frame,
        coarse_c1_values,
        coarse_c2_values,
        theta_true,
        global_score_heatmap_path,
        f"{model_name.upper()} score test (global coarse audit): {dataset_name}",
        labels,
        {**heatmap_config, "zoom_to_contours": False},
        "q_score",
    )
    plot_heatmap(
        likelihood_frame,
        likelihood_c1_values,
        likelihood_c2_values,
        theta_true,
        likelihood_heatmap_path,
        f"{model_name.upper()} raw likelihood diagnostic: {dataset_name}",
        labels,
        heatmap_config,
        "q_relative",
    )
    return {
        "grid_results": str(table_path),
        "local_score_grid_results": str(score_table_path),
        "local_likelihood_grid_results": str(likelihood_table_path),
        "q_score_heatmap": str(score_heatmap_path),
        "q_score_global_coarse_heatmap": str(global_score_heatmap_path),
        "q_likelihood_raw_heatmap": str(likelihood_heatmap_path),
    }


def manifest_summary(
    theta_true: ThetaPoint,
    frame: pd.DataFrame,
    q_frame: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    event_count: int,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    plot_output_dir: Path,
) -> dict[str, Any]:
    raw_best = frame.loc[frame["minus2_sum_log_r"].idxmin()]
    calibrated_best = frame.loc[frame["sum_log_r_calibrated"].idxmax()]
    return {
        "model": model_name,
        "dataset": dataset_name,
        "theta_true": [theta_true.c1, theta_true.c2],
        "best_grid_point": [float(raw_best["c1"]), float(raw_best["c2"])],
        "best_calibrated_grid_point": [float(calibrated_best["c1"]), float(calibrated_best["c2"])],
        "best_error_aware_grid_point": [diagnostics["best_error_aware_c1"], diagnostics["best_error_aware_c2"]],
        "best_score_grid_point": [diagnostics["best_score_c1"], diagnostics["best_score_c2"]],
        "truth_q_relative": diagnostics["truth_q_relative"],
        "truth_q_calibrated": diagnostics.get("truth_q_calibrated", float("nan")),
        "truth_q_error_aware": diagnostics["truth_q_error_aware"],
        "truth_q_score": diagnostics["truth_q_score"],
        "truth_delta_sum_log_r_calibrated_per_event": diagnostics["truth_delta_sum_log_r_calibrated_per_event"],
        "n_events": int(event_count),
        "grid_points": int(len(frame)),
        "score_calibration": str(score_calibration_output_path(output_dir, model_name)),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Constraint/heatmap JSON config.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to config output_dir.")
    parser.add_argument("--plot-output-dir", default=None, help="Plot directory. Defaults to config plot_output_dir.")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to run, e.g. rascal alices.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names to run, e.g. c1_p10_c2_p0.")
    parser.add_argument("--max-events", type=int, default=None, help="Maximum events per dataset. Default: heatmap.max_events.")
    parser.add_argument("--c1-bins", type=int, default=None, help="Override theta_grid.c1_bins.")
    parser.add_argument("--c2-bins", type=int, default=None, help="Override theta_grid.c2_bins.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override event inference batch size.")
    parser.add_argument("--theta-batch-size", type=int, default=None, help="Number of theta grid points evaluated together.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    results = run_constraints(parse_args(argv))
    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
