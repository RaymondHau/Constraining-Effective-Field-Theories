#!/usr/bin/env python3
"""Run resumable Optuna scans of the augmented-loss alpha for each estimator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import optuna
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "WBF" / "neural_training.json"
DEFAULT_OUTPUT = REPO_ROOT / "table_outputs" / "optuna_alpha_scan"
DEFAULT_PLOT_OUTPUT = REPO_ROOT / "plotting_outputs" / "optuna_alpha_scan"
ALPHA_RANGES = {
    "RASCAL": (1.0, 500.0),
    "CASCAL": (0.05, 50.0),
    "ALICES": (0.01, 10.0),
}
PAPER_ALPHAS = {"RASCAL": 100.0, "CASCAL": 5.0, "ALICES": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", default=["RASCAL", "CASCAL", "ALICES"])
    parser.add_argument("--trials-per-method", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--metric",
        choices=["marginal_bce", "rmse", "trimmed_rmse"],
        default="marginal_bce",
    )
    parser.add_argument("--trim-quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--plot-output-dir", type=Path, default=DEFAULT_PLOT_OUTPUT)
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.resolve().open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def configured_alpha(config: dict[str, Any], method: str) -> float | None:
    for entry in config.get("training", {}).get("methods", []):
        if str(entry.get("name", entry.get("method", ""))).upper() == method:
            return float(entry["alpha"])
    return None


def trial_config(
    base_config: dict[str, Any],
    method: str,
    alpha: float,
    epochs: int,
    trim_quantile: float,
    trial_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("training", {})["methods"] = [{"name": method, "alpha": alpha}]
    training_config = config["training"].setdefault("training_config", {})
    training_config.update({
        "epochs": epochs,
        "patience": epochs + 1,
        "group_ratio_methods": False,
        "optuna_scan_mode": True,
        "optuna_trim_quantile": trim_quantile,
    })
    paths = config.setdefault("paths", {})
    paths["model_subdir"] = str((trial_dir / "models").resolve())
    paths["performance_plot_subdir"] = str((trial_dir / "plots").resolve())
    config["name"] = f"optuna_{method.lower()}_alpha_{alpha:.8g}"
    return config


def run_trial(
    trial: optuna.Trial,
    method: str,
    base_config: dict[str, Any],
    output_dir: Path,
    epochs: int,
    metric: str,
    trim_quantile: float,
) -> float:
    low, high = ALPHA_RANGES[method]
    alpha = trial.suggest_float("alpha", low, high, log=True)
    trial_dir = output_dir / "trials" / method.lower() / metric / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "neural_training.json"
    config_path.write_text(
        json.dumps(trial_config(base_config, method, alpha, epochs, trim_quantile, trial_dir), indent=2),
        encoding="utf-8",
    )

    log_path = trial_dir / "training.log"
    env = os.environ.copy()
    env["EFT_WORKFLOW_CONFIG"] = str(config_path.resolve())
    command = [sys.executable, "-u", str(REPO_ROOT / "scripts" / "EFT_train_estimators.py")]
    print(f"[optuna] {method} trial={trial.number} alpha={alpha:.8g}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {completed.returncode}; see {log_path}")

    metrics_path = trial_dir / "models" / "validation_marginal_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    row = metrics.loc[metrics["method"].astype(str).str.upper() == method]
    if len(row) != 1:
        raise RuntimeError(f"Expected one {method} row in {metrics_path}, found {len(row)}")
    result = row.iloc[0]
    metric_columns = [
        "marginal_bce", "brier",
        "rmse", "trimmed_rmse", "mae", "bias", "corr", "y0_rmse", "y1_rmse",
        "evaluation_rows", "theta_points",
    ]
    for column in metric_columns:
        if column in result and pd.notna(result[column]):
            trial.set_user_attr(column, float(result[column]))
    trial.set_user_attr("metrics_path", str(metrics_path))
    value = float(result[metric])
    print(f"[optuna] {method} trial={trial.number} {metric}={value:.8g}", flush=True)
    return value


def enqueue_baselines(study: optuna.Study, base_config: dict[str, Any], method: str) -> None:
    if study.trials:
        return
    values = [configured_alpha(base_config, method), PAPER_ALPHAS.get(method)]
    seen = set()
    for value in values:
        if value is not None and ALPHA_RANGES[method][0] <= value <= ALPHA_RANGES[method][1] and value not in seen:
            study.enqueue_trial({"alpha": value})
            seen.add(value)


def write_summaries(
    studies: dict[str, optuna.Study],
    base_config: dict[str, Any],
    output_dir: Path,
    metric: str,
) -> None:
    summary_rows = []
    best_config = copy.deepcopy(base_config)
    best_methods = []
    for method, study in studies.items():
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            continue
        best = study.best_trial
        summary_rows.append({
            "method": method,
            "best_alpha": float(best.params["alpha"]),
            "objective": metric,
            "best_value": float(best.value),
            "completed_trials": len(completed),
            **{key: best.user_attrs.get(key) for key in [
                "marginal_bce", "brier",
                "rmse", "trimmed_rmse", "mae", "bias", "corr", "y0_rmse", "y1_rmse",
            ]},
        })
        best_methods.append({"name": method, "alpha": float(best.params["alpha"])})
        study.trials_dataframe().to_csv(output_dir / f"{method.lower()}_trials.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(output_dir / "best_alpha_summary.csv", index=False)
    best_config.setdefault("training", {})["methods"] = best_methods
    (output_dir / "best_alpha_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    (output_dir / "best_alphas.json").write_text(
        json.dumps({row["method"]: row["best_alpha"] for row in summary_rows}, indent=2),
        encoding="utf-8",
    )


def prune_non_best_checkpoints(
    study: optuna.Study,
    output_dir: Path,
    method: str,
    metric: str,
) -> None:
    """Keep only the current best trial's model checkpoint."""
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    best_metrics_path = study.best_trial.user_attrs.get("metrics_path")
    if not best_metrics_path:
        return
    best_model_dir = Path(str(best_metrics_path)).resolve().parent
    trial_root = output_dir / "trials" / method.lower() / metric
    for model_dir in trial_root.glob("trial_*/models"):
        if model_dir.resolve() == best_model_dir:
            continue
        for pattern in ("*.pt",):
            for artifact in model_dir.glob(pattern):
                artifact.unlink(missing_ok=True)


def completed_trials_table(path: Path) -> tuple[str, pd.DataFrame]:
    """Load finite completed trials from an exported Optuna table."""
    method = path.stem.removesuffix("_trials").upper()
    trials = pd.read_csv(path)
    required = {"number", "params_alpha", "state", "value"}
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    trials = trials.loc[trials["state"].astype(str).str.upper() == "COMPLETE"].copy()
    trials["alpha"] = pd.to_numeric(trials["params_alpha"], errors="coerce")
    trials["objective"] = pd.to_numeric(trials["value"], errors="coerce")
    trials = trials.replace([np.inf, -np.inf], np.nan).dropna(subset=["alpha", "objective"])
    if trials.empty:
        raise ValueError(f"{path} contains no finite completed trials")
    return method, trials


def resolve_best_checkpoint(scan_dir: Path, trials_path: Path, method: str, best: pd.Series) -> Path:
    """Resolve a best checkpoint after results have been copied from CSD3."""
    metrics_path = best.get("user_attrs_metrics_path")
    if isinstance(metrics_path, str) and metrics_path:
        local_metrics = Path(metrics_path)
        marker = (Path("table_outputs") / "optuna_alpha_scan").as_posix()
        if not local_metrics.exists() and marker in local_metrics.as_posix():
            local_metrics = scan_dir / local_metrics.as_posix().split(marker, 1)[1].lstrip("/")
        checkpoint = local_metrics.parent / f"{method.lower()}.pt"
        if checkpoint.exists():
            return checkpoint
    trial_number = int(best["number"])
    candidates = list(
        (trials_path.parent / "trials" / method.lower()).glob(
            f"**/trial_{trial_number:04d}/models/{method.lower()}.pt"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one checkpoint for {method} trial {trial_number}, found {candidates}")
    return candidates[0]


def plot_alpha_trials(ax: plt.Axes, method: str, trials: pd.DataFrame) -> pd.Series:
    """Plot objective against alpha and return the winning trial."""
    ordered = trials.sort_values("alpha")
    best = trials.loc[trials["objective"].idxmin()]
    ax.scatter(ordered["alpha"], ordered["objective"], s=32, alpha=0.75, label="Completed trial")
    ax.scatter(
        [best["alpha"]], [best["objective"]], marker="*", s=180, color="crimson",
        edgecolor="black", linewidth=0.5, zorder=3, label=f"Best alpha={best['alpha']:.4g}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Alpha")
    if "user_attrs_marginal_bce" in trials and np.allclose(trials["objective"], trials["user_attrs_marginal_bce"]):
        ax.set_ylabel("Marginal BCE")
    else:
        ax.set_ylabel("Optuna objective")
    ax.set_title(method)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return best


def checkpoint_activation(torch, name: str):
    choices = {
        "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU, "swish": torch.nn.SiLU,
    }
    if name.lower() not in choices:
        raise ValueError(f"Unsupported checkpoint activation: {name}")
    return choices[name.lower()]()


def build_checkpoint_model(torch, checkpoint: dict, input_dim: int):
    """Reconstruct the ratio network stored by EFT_train_estimators.py."""
    training = checkpoint["training_config"]
    layers = []
    previous = input_dim
    for width in map(int, training["hidden_layers"]):
        layers.extend([torch.nn.Linear(previous, width), checkpoint_activation(torch, training.get("activation", "tanh"))])
        if float(training.get("dropout", 0.0)) > 0.0:
            layers.append(torch.nn.Dropout(float(training["dropout"])))
        previous = width
    layers.append(torch.nn.Linear(previous, 1))

    class RatioModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(*layers)

        def forward(self, features, theta):
            return self.network(torch.cat([features, theta], dim=1))

    model = RatioModel()
    model.load_state_dict(checkpoint["state_dict"])
    return model


def collect_checkpoint_predictions(
    checkpoint_path: Path,
    method: str,
    test: pd.DataFrame,
    feature_columns: list[str],
    operators: list[str],
    requested_device: str,
) -> pd.DataFrame:
    """Run one saved winning checkpoint on the ratio test sample."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for performance plots") from exc
    device = requested_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    scalers = checkpoint["scalers"]
    feature_mean = np.asarray(scalers["feature"]["mean"], dtype=np.float32)
    feature_scale = np.asarray(scalers["feature"]["scale"], dtype=np.float32)
    theta_mean = np.asarray(scalers["theta"]["mean"], dtype=np.float32)
    theta_scale = np.asarray(scalers["theta"]["scale"], dtype=np.float32)
    log_r_mean = np.asarray(scalers["log_r"]["mean"], dtype=np.float32)
    log_r_scale = np.asarray(scalers["log_r"]["scale"], dtype=np.float32)
    features = (test[feature_columns].to_numpy(dtype=np.float32) - feature_mean) / feature_scale
    theta_columns = [f"theta0_{name}" for name in operators]
    theta = (test[theta_columns].to_numpy(dtype=np.float32) - theta_mean) / theta_scale
    model = build_checkpoint_model(torch, checkpoint, features.shape[1] + theta.shape[1]).to(device).eval()
    log_r_parts, score_parts = [], []
    batch_size = int(checkpoint["training_config"].get("batch_size", 32768))
    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        feature_batch = torch.as_tensor(features[start:stop], device=device)
        theta_batch = torch.as_tensor(theta[start:stop], device=device).requires_grad_(True)
        scaled_log_r = model(feature_batch, theta_batch)
        log_r = scaled_log_r * torch.as_tensor(log_r_scale, device=device) + torch.as_tensor(log_r_mean, device=device)
        score = torch.autograd.grad(log_r.sum(), theta_batch)[0] / torch.as_tensor(theta_scale, device=device)
        log_r_parts.append(log_r.detach().cpu().numpy().ravel())
        score_parts.append(score.detach().cpu().numpy())
    result = pd.DataFrame({
        "method": method, "y": test["y"].to_numpy(), "log_r_true": test["log_r"].to_numpy(),
        "log_r_pred": np.concatenate(log_r_parts),
    })
    scores = np.concatenate(score_parts)
    for index, operator in enumerate(operators):
        result[f"theta0_{operator}"] = test[f"theta0_{operator}"].to_numpy()
        result[f"score_true_{operator}"] = test[f"score_{operator}"].to_numpy()
        result[f"score_pred_{operator}"] = scores[:, index]
    return result


def save_prediction_scatter(
    frame: pd.DataFrame,
    method: str,
    true_column: str,
    predicted_column: str,
    label: str,
    path: Path,
    max_points: int,
) -> None:
    """Save an unclipped true-vs-predicted scatter plot."""
    values = frame[[true_column, predicted_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) > max_points:
        values = values.sample(max_points, random_state=1234)
    low, high = float(values.to_numpy().min()), float(values.to_numpy().max())
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(values[true_column], values[predicted_column], s=8, alpha=0.3, linewidths=0)
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(method)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_winning_performance(best_rows: list[dict], config_path: Path, output_dir: Path, device: str) -> None:
    """Generate raw performance plots from only the winning checkpoints."""
    config = load_config(config_path)
    paths = config.get("paths", {})
    storage = Path(paths.get("storage_workspace", ".")).expanduser()
    if not storage.is_absolute():
        storage = REPO_ROOT / storage
    input_dir = storage / paths.get("table_subdir", "table_outputs") / paths.get("sample_output_subdir", "madminer_style_training")
    operators = list(config["physics"]["eft_operators"])
    feature_columns = list(config["physics"]["feature_columns"])
    required = [*feature_columns, *[f"theta0_{name}" for name in operators], "y", "log_r", *[f"score_{name}" for name in operators]]
    test = pd.read_csv(input_dir / "ratio_test.csv", usecols=required)
    performance_dir = output_dir / "best_performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    training = config.get("training", {}).get("training_config", {})
    max_points = int(training.get("plot_max_points", 50000))
    predictions, metrics = [], []
    for best in best_rows:
        method = str(best["method"])
        frame = collect_checkpoint_predictions(
            Path(str(best["checkpoint"])), method, test, feature_columns, operators, device
        )
        predictions.append(frame)
        y = frame["y"].to_numpy(dtype=np.float64)
        log_r = frame["log_r_pred"].to_numpy(dtype=np.float64)
        residual = log_r - frame["log_r_true"].to_numpy(dtype=np.float64)
        metrics.append({
            "method": method,
            "alpha": best["alpha"],
            "marginal_bce": float(np.mean(np.logaddexp(0.0, log_r) - y * log_r)),
            "joint_log_r_rmse_diagnostic": float(np.sqrt(np.mean(np.square(residual)))),
        })
        save_prediction_scatter(
            frame, method, "log_r_true", "log_r_pred", "joint log-r target (diagnostic)",
            performance_dir / f"{method.lower()}_log_r_scatter.png", max_points,
        )
        numerator = frame.loc[frame["y"] == 1.0]
        for operator in operators:
            save_prediction_scatter(
                numerator, method, f"score_true_{operator}", f"score_pred_{operator}", f"score {operator}",
                performance_dir / f"{method.lower()}_score_{operator}_scatter.png", max_points,
            )
    combined = pd.concat(predictions, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method, frame in combined.groupby("method"):
        residual = (frame["log_r_pred"] - frame["log_r_true"]).replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(residual, bins=80, histtype="step", density=True, label=method)
    ax.set_xlabel("Marginal prediction - joint log-r target")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(performance_dir / "log_r_residuals.png", dpi=180)
    plt.close(fig)
    for operator in operators:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for method, frame in combined.loc[combined["y"] == 1.0].groupby("method"):
            residual = (frame[f"score_pred_{operator}"] - frame[f"score_true_{operator}"]).replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(residual, bins=80, histtype="step", density=True, label=method)
        ax.set_xlabel(f"Predicted - true score {operator}")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(performance_dir / f"score_{operator}_residuals.png", dpi=180)
        plt.close(fig)
    pd.DataFrame(metrics).to_csv(performance_dir / "best_test_metrics.csv", index=False)


def plot_saved_scan(
    scan_dir: Path,
    plot_output_dir: Path,
    config_path: Path,
    skip_performance: bool,
    device: str,
) -> None:
    """Plot completed studies and evaluate only their winning checkpoints."""
    scan_dir, plot_output_dir = scan_dir.resolve(), plot_output_dir.resolve()
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = sorted(scan_dir.rglob("*_trials.csv"))
    if not trial_paths:
        raise FileNotFoundError(f"No *_trials.csv files found below {scan_dir}")
    scans = [(method, trials, path) for path in trial_paths for method, trials in [completed_trials_table(path)]]
    fig, axes = plt.subplots(1, len(scans), figsize=(6 * len(scans), 4.8), squeeze=False)
    best_rows = []
    for ax, (method, trials, trials_path) in zip(axes[0], scans):
        best = plot_alpha_trials(ax, method, trials)
        checkpoint = resolve_best_checkpoint(scan_dir, trials_path, method, best)
        best_rows.append({
            "method": method,
            "trial": int(best["number"]),
            "alpha": float(best["alpha"]),
            "objective": float(best["objective"]),
            "checkpoint": str(checkpoint),
        })
    fig.tight_layout()
    objective_name = "marginal_bce" if all(
        "user_attrs_marginal_bce" in trials and np.allclose(trials["objective"], trials["user_attrs_marginal_bce"])
        for _, trials, _ in scans
    ) else "objective"
    fig.savefig(plot_output_dir / f"alpha_vs_{objective_name}.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(best_rows).drop(columns="checkpoint").to_csv(
        plot_output_dir / "best_alpha_plot_summary.csv", index=False
    )
    if not skip_performance:
        plot_winning_performance(best_rows, config_path, plot_output_dir, device)
    print(f"Wrote Optuna plots to {plot_output_dir}")


def main() -> int:
    args = parse_args()
    if args.plot_only:
        plot_saved_scan(
            args.output_dir,
            args.plot_output_dir,
            args.config,
            args.skip_performance,
            args.device,
        )
        return 0
    if args.trials_per_method <= 0 or args.epochs <= 0:
        raise ValueError("trials-per-method and epochs must be positive")
    if not 0.0 < args.trim_quantile <= 1.0:
        raise ValueError("trim-quantile must lie in (0, 1]")
    methods = [method.upper() for method in args.methods]
    unsupported = sorted(set(methods) - set(ALPHA_RANGES))
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.config)
    config_fingerprint = hashlib.sha256(
        json.dumps(base_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    storage = f"sqlite:///{(output_dir / 'optuna_alpha.db').resolve()}"
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    studies = {}
    for method in methods:
        study = optuna.create_study(
            study_name=f"alpha_{method.lower()}_{args.metric}_{args.epochs}ep_{config_fingerprint}",
            storage=storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
        enqueue_baselines(study, base_config, method)
        completed_count = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        remaining = max(0, args.trials_per_method - completed_count)
        if remaining:
            study.optimize(
                lambda trial, selected=method: run_trial(
                    trial,
                    selected,
                    base_config,
                    output_dir,
                    args.epochs,
                    args.metric,
                    args.trim_quantile,
                ),
                n_trials=remaining,
                catch=(RuntimeError,),
                callbacks=[
                    lambda current_study, _trial, selected=method: prune_non_best_checkpoints(
                        current_study, output_dir, selected, args.metric
                    )
                ],
            )
        prune_non_best_checkpoints(study, output_dir, method, args.metric)
        studies[method] = study
        write_summaries(studies, base_config, output_dir, args.metric)
        completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            print(f"[optuna] {method} best alpha={study.best_params['alpha']:.8g}, {args.metric}={study.best_value:.8g}", flush=True)
        else:
            print(f"[optuna] {method} has no completed trials; inspect trial logs under {output_dir}", flush=True)

    write_summaries(studies, base_config, output_dir, args.metric)
    print(f"[optuna] summaries written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
