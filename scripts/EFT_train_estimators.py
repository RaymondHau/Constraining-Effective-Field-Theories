#!/usr/bin/env python
# coding: utf-8

# # Train MadMiner-style Neural Estimators
# 
# This notebook consumes the ratio and local-score CSVs written by `EFT_prepare_madminer_style_samples.ipynb`.
# 
# It trains RASCAL, CASCAL, ALICES, and SALLINO estimators, saves model checkpoints, and makes performance plots on the held-out test split.

# ## 1. Imports

# In[34]:


from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

try:
    from IPython.display import display
except Exception:
    def display(value):
        print(value)

plt.rcParams.update({"figure.figsize": (7, 4), "axes.grid": True})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# ## 2. Configuration

# In[35]:


NOTEBOOK_DIR = Path.cwd()
LOCAL_WORKSPACE_DIR = NOTEBOOK_DIR / "barebones_eft_workspace"
def preferred_storage_workspace() -> Path:
    """Use D: for bulky tables/models when it is mounted, otherwise stay local."""
    default_external = Path.cwd()
    requested = Path(os.environ.get("EFT_STORAGE_WORKSPACE", str(default_external))).expanduser()
    if requested.parent.exists() and requested.parent.is_dir():
        try:
            requested.mkdir(parents=True, exist_ok=True)
            probe = requested / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return requested
        except OSError as exc:
            print(f"External storage unavailable at {requested}: {exc}. Using local workspace.")
    return Path.cwd() / "barebones_eft_workspace"

STORAGE_WORKSPACE_DIR = preferred_storage_workspace()
WORKSPACE_DIR = STORAGE_WORKSPACE_DIR
TABLE_DIR = WORKSPACE_DIR / "tables"
INPUT_DIR = TABLE_DIR / "madminer_style_training"
MODEL_DIR = INPUT_DIR / "trained_estimators"
PLOT_DIR = INPUT_DIR / "performance_plots"

_PATH_CONFIG = {}
if os.environ.get("EFT_WORKFLOW_CONFIG"):
    try:
        with open(os.environ["EFT_WORKFLOW_CONFIG"], "r", encoding="utf-8-sig") as handle:
            _PATH_CONFIG = json.load(handle).get("paths", {})
    except OSError:
        _PATH_CONFIG = {}
if _PATH_CONFIG:
    if "storage_workspace" in _PATH_CONFIG:
        STORAGE_WORKSPACE_DIR = Path(_PATH_CONFIG["storage_workspace"]).expanduser()
        STORAGE_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR = STORAGE_WORKSPACE_DIR
    TABLE_DIR = WORKSPACE_DIR / _PATH_CONFIG.get("table_subdir", "tables")
    INPUT_DIR = TABLE_DIR / _PATH_CONFIG.get("sample_output_subdir", "madminer_style_training")
    MODEL_DIR = INPUT_DIR / _PATH_CONFIG.get("model_subdir", "trained_estimators")
    PLOT_DIR = INPUT_DIR / _PATH_CONFIG.get("performance_plot_subdir", "performance_plots")

for folder in [MODEL_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["pt_j1", "delta_phi_jj", "met"]
EFT_OPERATORS = ["CWL2", "CPWL2"]
SPLITS = ["train", "validation", "test"]
METHODS = ["RASCAL", "CASCAL", "ALICES"]

TRAINING_CONFIG = {
    "batch_size": 4096,
    "epochs": 200,
    "learning_rate": 0.0015,
    "min_learning_rate": 1.0e-5,
    "weight_decay": 0, # 0.25,
    "hidden_layers": [1024, 1024, 1024, 1024, 1024],    #[1024, 1024, 256, 128]
    "dropout": 0.0,
    "feature_noise_std": 0.0,
    "alpha": 0.10,
    "gradient_clip": 1000000.0,
    "patience": 200,
    "min_delta": 1.0e-6,
    "seed": 1234,
    "activation": "tanh",
    "csv_chunk_rows": 100_000,
}


def load_workflow_config() -> Dict:
    config_path = os.environ.get("EFT_WORKFLOW_CONFIG")
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


WORKFLOW_CONFIG = load_workflow_config()


def load_stage_config(stage_name: str) -> Dict:
    return WORKFLOW_CONFIG.get("stage_config", {}).get(stage_name, {})


_STAGE_CONFIG = load_stage_config("train_estimators")
_PHYSICS_CONFIG = WORKFLOW_CONFIG.get("physics", {})
_TRAINING_SECTION = WORKFLOW_CONFIG.get("training", {})
if "feature_columns" in _PHYSICS_CONFIG:
    FEATURE_COLUMNS = list(_PHYSICS_CONFIG["feature_columns"])
if "eft_operators" in _PHYSICS_CONFIG:
    EFT_OPERATORS = list(_PHYSICS_CONFIG["eft_operators"])
if "methods" in _TRAINING_SECTION:
    METHODS = list(_TRAINING_SECTION["methods"])
if "methods" in _STAGE_CONFIG:
    METHODS = list(_STAGE_CONFIG["methods"])
if "training_config" in _TRAINING_SECTION:
    TRAINING_CONFIG.update(_TRAINING_SECTION["training_config"])
if "training_config" in _STAGE_CONFIG:
    TRAINING_CONFIG.update(_STAGE_CONFIG["training_config"])
print("Training methods:", METHODS)
print("Training config:", TRAINING_CONFIG)


# ## 3. Load Prepared Samples

# In[36]:


def read_required_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load a required CSV and give a clear error when it is missing."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=usecols)


ratio_paths = {split: INPUT_DIR / f"ratio_{split}.csv" for split in SPLITS}
local_paths = {split: INPUT_DIR / f"local_{split}.csv" for split in SPLITS}

required_ratio = ["split", "event_id", *FEATURE_COLUMNS, "y", "soft_y", "log_r", "likelihood_ratio"]
required_local = ["split", "event_id", *FEATURE_COLUMNS]

for name in EFT_OPERATORS:
    required_ratio += [f"theta0_{name}", f"theta1_{name}", f"score_{name}"]
    required_local += [f"theta_{name}", f"score_{name}"]

def validate_required_columns(path: Path, required_columns: list[str]) -> None:
    """Check CSV headers without loading the table."""
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    missing = [column for column in required_columns if column not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


for split in SPLITS:
    validate_required_columns(ratio_paths[split], required_ratio)
    validate_required_columns(local_paths[split], required_local)

ratio_frames = {split: read_required_csv(ratio_paths[split], usecols=required_ratio) for split in ["validation", "test"]}
local_frames = {split: read_required_csv(local_paths[split], usecols=required_local) for split in ["validation", "test"]}

summary_path = INPUT_DIR / "sample_summary.csv"
if summary_path.exists():
    sample_summary = pd.read_csv(summary_path)
    print("Prepared sample rows:")
    display(sample_summary)
else:
    print("Prepared sample row summary not found; train CSVs will be streamed without pre-counting.")
print("Loaded validation/test ratio rows:", {split: len(frame) for split, frame in ratio_frames.items()})
print("Loaded validation/test local rows:", {split: len(frame) for split, frame in local_frames.items()})


# ## 4. Datasets and Normalization

# In[37]:


@dataclass
class Standardizer:
    """Mean and scale for standardizing arrays."""
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        # Shift by mean and scale to standardize input values
        return (values - self.mean) / self.scale


def fit_standardizer(values: np.ndarray) -> Standardizer:
    """Fit a standardizer on training values."""
    # Compute mean and standard deviation along axis 0
    mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    scale = values.std(axis=0, keepdims=True).astype(np.float32)
    # Avoid division by zero if variance is zero
    scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
    return Standardizer(mean, scale)


theta0_columns = [f"theta0_{name}" for name in EFT_OPERATORS]
theta_columns = [f"theta_{name}" for name in EFT_OPERATORS]
score_columns = [f"score_{name}" for name in EFT_OPERATORS]
CSV_CHUNK_ROWS = int(TRAINING_CONFIG.get("csv_chunk_rows", 100_000))


def fit_standardizer_from_csv(sources: list[Tuple[Path, list[str]]]) -> Standardizer:
    """Fit a standardizer by streaming selected columns from one or more CSV files."""
    total_count = 0
    total_sum = None
    total_sumsq = None
    for path, columns in sources:
        print(f"Fitting scaler from {path.name}: {columns}", flush=True)
        for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=columns, chunksize=CSV_CHUNK_ROWS), start=1):
            values = chunk[columns].to_numpy(dtype=np.float64)
            if total_sum is None:
                total_sum = np.zeros(values.shape[1], dtype=np.float64)
                total_sumsq = np.zeros(values.shape[1], dtype=np.float64)
            total_count += len(values)
            total_sum += values.sum(axis=0)
            total_sumsq += np.square(values).sum(axis=0)
            if chunk_index == 1 or chunk_index % 10 == 0:
                print(f"  {path.name}: scaler rows processed {total_count:,}", flush=True)
    if total_count == 0 or total_sum is None or total_sumsq is None:
        raise RuntimeError("Cannot fit scaler from empty CSV input")
    mean_1d = total_sum / total_count
    variance_1d = np.maximum(total_sumsq / total_count - np.square(mean_1d), 0.0)
    scale_1d = np.sqrt(variance_1d)
    scale_1d = np.where(scale_1d > 0.0, scale_1d, 1.0)
    return Standardizer(mean_1d[None, :].astype(np.float32), scale_1d[None, :].astype(np.float32))


feature_scaler = fit_standardizer_from_csv([(ratio_paths["train"], FEATURE_COLUMNS)])
theta_scaler = fit_standardizer_from_csv([(ratio_paths["train"], theta0_columns)])
log_r_scaler = fit_standardizer_from_csv([(ratio_paths["train"], ["log_r"])])
score_scaler = fit_standardizer_from_csv([(ratio_paths["train"], score_columns), (local_paths["train"], score_columns)])


class RatioDataset(Dataset):
    """Dataset for RASCAL, CASCAL, and ALICES."""
    def __init__(self, frame: pd.DataFrame):
        """Convert a ratio sample DataFrame to tensors."""
        # Scale and convert input features and physics parameters (theta)
        self.features = torch.as_tensor(feature_scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)), dtype=torch.float32)
        self.theta = torch.as_tensor(theta_scaler.transform(frame[theta0_columns].to_numpy(dtype=np.float32)), dtype=torch.float32)

        # Load targets: exact target labels y, soft targets, and true log-ratios
        self.y = torch.as_tensor(frame[["y"]].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.soft_y = torch.as_tensor(frame[["soft_y"]].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.log_r = torch.as_tensor(frame[["log_r"]].to_numpy(dtype=np.float32), dtype=torch.float32)

        # Scale log-ratios and raw scores for standard training losses
        self.log_r_scaled = torch.as_tensor(log_r_scaler.transform(frame[["log_r"]].to_numpy(dtype=np.float32)), dtype=torch.float32)
        self.score = torch.as_tensor(frame[score_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.score_scaled = torch.as_tensor(score_scaler.transform(frame[score_columns].to_numpy(dtype=np.float32)), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one ratio-training row."""
        # Package and return a single event batch item
        keys = ["features", "theta", "y", "soft_y", "log_r", "log_r_scaled", "score", "score_scaled"]
        return {key: getattr(self, key)[index] for key in keys}


class LocalScoreDataset(Dataset):
    """Dataset for SALLINO direct-score training."""
    def __init__(self, frame: pd.DataFrame):
        """Convert a local-score sample DataFrame to tensors."""
        # Scale features and local physical parameters
        self.features = torch.as_tensor(feature_scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)), dtype=torch.float32)
        self.theta = torch.as_tensor(theta_scaler.transform(frame[theta_columns].to_numpy(dtype=np.float32)), dtype=torch.float32)

        # Load true local score components and apply standardization
        self.score = torch.as_tensor(frame[score_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.score_scaled = torch.as_tensor(score_scaler.transform(frame[score_columns].to_numpy(dtype=np.float32)), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one local-score row."""
        # Fetch properties as dictionary values
        return {key: getattr(self, key)[index] for key in ["features", "theta", "score", "score_scaled"]}


class RatioCSVBatchDataset(IterableDataset):
    """Stream ratio training batches from CSV chunks."""
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        batch_size = int(TRAINING_CONFIG["batch_size"])
        for chunk_index, frame in enumerate(pd.read_csv(self.path, usecols=required_ratio, chunksize=CSV_CHUNK_ROWS), start=1):
            if chunk_index == 1 or chunk_index % 10 == 0:
                print(f"Streaming {self.path.name} chunk {chunk_index} ({len(frame):,} rows)", flush=True)
            payload = {
                "features": torch.as_tensor(feature_scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)), dtype=torch.float32),
                "theta": torch.as_tensor(theta_scaler.transform(frame[theta0_columns].to_numpy(dtype=np.float32)), dtype=torch.float32),
                "y": torch.as_tensor(frame[["y"]].to_numpy(dtype=np.float32), dtype=torch.float32),
                "soft_y": torch.as_tensor(frame[["soft_y"]].to_numpy(dtype=np.float32), dtype=torch.float32),
                "log_r": torch.as_tensor(frame[["log_r"]].to_numpy(dtype=np.float32), dtype=torch.float32),
                "log_r_scaled": torch.as_tensor(log_r_scaler.transform(frame[["log_r"]].to_numpy(dtype=np.float32)), dtype=torch.float32),
                "score": torch.as_tensor(frame[score_columns].to_numpy(dtype=np.float32), dtype=torch.float32),
                "score_scaled": torch.as_tensor(score_scaler.transform(frame[score_columns].to_numpy(dtype=np.float32)), dtype=torch.float32),
            }
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                yield {key: value[start:stop] for key, value in payload.items()}


class LocalCSVBatchDataset(IterableDataset):
    """Stream local-score training batches from CSV chunks."""
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        batch_size = int(TRAINING_CONFIG["batch_size"])
        for chunk_index, frame in enumerate(pd.read_csv(self.path, usecols=required_local, chunksize=CSV_CHUNK_ROWS), start=1):
            if chunk_index == 1 or chunk_index % 10 == 0:
                print(f"Streaming {self.path.name} chunk {chunk_index} ({len(frame):,} rows)", flush=True)
            payload = {
                "features": torch.as_tensor(feature_scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)), dtype=torch.float32),
                "theta": torch.as_tensor(theta_scaler.transform(frame[theta_columns].to_numpy(dtype=np.float32)), dtype=torch.float32),
                "score": torch.as_tensor(frame[score_columns].to_numpy(dtype=np.float32), dtype=torch.float32),
                "score_scaled": torch.as_tensor(score_scaler.transform(frame[score_columns].to_numpy(dtype=np.float32)), dtype=torch.float32),
            }
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                yield {key: value[start:stop] for key, value in payload.items()}


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    """Return a DataLoader with the configured batch size."""
    # Iterable CSV datasets yield already-batched dictionaries.
    if isinstance(dataset, IterableDataset):
        return DataLoader(dataset, batch_size=None)
    return DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=shuffle, drop_last=False)


ratio_loaders = {
    "train": make_loader(RatioCSVBatchDataset(ratio_paths["train"]), shuffle=False),
    **{split: make_loader(RatioDataset(frame), shuffle=False) for split, frame in ratio_frames.items()},
}
local_loaders = {
    "train": make_loader(LocalCSVBatchDataset(local_paths["train"]), shuffle=False),
    **{split: make_loader(LocalScoreDataset(frame), shuffle=False) for split, frame in local_frames.items()},
}


# ## 5. Model Definitions

# In[38]:


def activation_layer() -> nn.Module:
    """Return the configured hidden-layer activation."""
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
    """Build a fully connected network with optional dropout."""
    layers = []
    previous = input_dim
    dropout = float(TRAINING_CONFIG.get("dropout", 0.0))
    # Iterate through hidden widths to add linear and activation layers
    for width in hidden_layers:
        layers.extend([nn.Linear(previous, width), activation_layer()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous = width
    # Project last layer to target output dimensionality
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class RatioEstimator(nn.Module):
    """Parameterized log-ratio estimator."""
    def __init__(self, input_dim: int, hidden_layers: Iterable[int]):
        """Initialize the ratio network."""
        super().__init__()
        # MLP architecture maps concatenated inputs to a single scale log-ratio
        self.network = build_mlp(input_dim, 1, hidden_layers)

    def forward(self, features: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Predict physical log r(theta, theta_ref)."""
        # Concat features and theta, run MLP, then scale back to physical units
        scaled = self.network(torch.cat([features, theta], dim=1))
        mean = torch.as_tensor(log_r_scaler.mean, dtype=torch.float32, device=scaled.device)
        scale = torch.as_tensor(log_r_scaler.scale, dtype=torch.float32, device=scaled.device)
        return scaled * scale + mean


class ScoreEstimator(nn.Module):
    """Parameterized direct score estimator."""
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Iterable[int]):
        """Initialize the score network."""
        super().__init__()
        # MLP maps inputs to score dimensions (number of operators)
        self.network = build_mlp(input_dim, output_dim, hidden_layers)

    def forward(self, features: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Predict physical score components directly."""
        # Concatenate features and parameters, then predict score coordinates
        return self.network(torch.cat([features, theta], dim=1))


def ratio_score_from_gradient(model: RatioEstimator, features: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return predicted log ratio and its theta-gradient score."""
    # Enable gradient tracking on theta inputs
    theta = theta.detach().clone().requires_grad_(True)
    # Predict log ratio
    log_r = model(features, theta)
    # Take derivative of predicted log ratio w.r.t theta to calculate score
    grad_scaled = torch.autograd.grad(log_r.sum(), theta, create_graph=model.training)[0]
    # Adjust for theta scale preprocessing standardizer
    theta_scale = torch.as_tensor(theta_scaler.scale, dtype=torch.float32, device=features.device)
    return log_r, grad_scaled / theta_scale


# ## 6. Losses and Training Loop

# In[39]:


def batch_to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Move batch tensors to DEVICE."""
    # Move key-value dictionary tensors to target execution device
    return {key: value.to(DEVICE) for key, value in batch.items()}


def scaled_log_r_tensor(values: torch.Tensor) -> torch.Tensor:
    """Convert physical log-ratio tensors to the standardized training scale."""
    # Standardize physical values for computing normalized loss functions
    mean = torch.as_tensor(log_r_scaler.mean, dtype=torch.float32, device=values.device)
    scale = torch.as_tensor(log_r_scaler.scale, dtype=torch.float32, device=values.device)
    return (values - mean) / scale


def scaled_score_tensor(values: torch.Tensor) -> torch.Tensor:
    """Convert physical score tensors to the standardized training scale."""
    # Standardize score components to standard normal target space
    mean = torch.as_tensor(score_scaler.mean, dtype=torch.float32, device=values.device)
    scale = torch.as_tensor(score_scaler.scale, dtype=torch.float32, device=values.device)
    return (values - mean) / scale


def ratio_loss(method: str, model: RatioEstimator, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return the method-specific ratio loss for one batch."""
    # Predict log r and its gradient-based score
    log_r_pred, score_pred = ratio_score_from_gradient(model, batch["features"], batch["theta"])

    # Compute score MSE loss in the standardized training scale
    score_pred_scaled = scaled_score_tensor(score_pred)
    score_loss = nn.functional.mse_loss(score_pred_scaled, batch["score_scaled"])

    # Calculate classifier-style main loss or regression loss depending on method
    if method == "ALICES":
        main_loss = nn.functional.binary_cross_entropy_with_logits(log_r_pred, batch["soft_y"])
    elif method == "CASCAL":
        main_loss = nn.functional.binary_cross_entropy_with_logits(log_r_pred, batch["y"])
    elif method == "RASCAL":
        main_loss = nn.functional.mse_loss(scaled_log_r_tensor(log_r_pred), batch["log_r_scaled"])
    else:
        raise ValueError(method)

    # Scale score loss weight by configuration alpha and return combined loss
    total = main_loss + TRAINING_CONFIG["alpha"] * score_loss
    return total, {"loss": float(total.detach().cpu()), "main_loss": float(main_loss.detach().cpu()), "score_loss": float(score_loss.detach().cpu())}


def sallino_loss(model: ScoreEstimator, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return the SALLINO direct score loss."""
    # Feed features and theta, compute prediction score
    score_pred = model(batch["features"], batch["theta"])
    # Return MSE loss between normalized prediction score and targets
    loss = nn.functional.mse_loss(scaled_score_tensor(score_pred), batch["score_scaled"])
    return loss, {"loss": float(loss.detach().cpu()), "main_loss": math.nan, "score_loss": float(loss.detach().cpu())}


def train_method(method: str) -> Tuple[nn.Module, pd.DataFrame]:
    """Train one estimator and return the best model plus history."""
    # Fix execution seeds for reproducibility
    torch.manual_seed(TRAINING_CONFIG["seed"])
    input_dim = len(FEATURE_COLUMNS) + len(EFT_OPERATORS)

    # Initialize appropriate estimator type
    if method == "SALLINO":
        model = ScoreEstimator(input_dim, len(EFT_OPERATORS), TRAINING_CONFIG["hidden_layers"]).to(DEVICE)
        loaders = local_loaders
    else:
        model = RatioEstimator(input_dim, TRAINING_CONFIG["hidden_layers"]).to(DEVICE)
        loaders = ratio_loaders

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"], weight_decay=TRAINING_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG["epochs"],
        eta_min=TRAINING_CONFIG["min_learning_rate"],
    )

    print(f"{method}: {sum(isinstance(module, nn.Dropout) for module in model.modules())} dropout layers, p={TRAINING_CONFIG['dropout']}")
    history = []
    best_state = None
    best_val = float("inf")
    stale_epochs = 0

    # Start training epoch loop
    for epoch in range(1, TRAINING_CONFIG["epochs"] + 1):
        model.train()
        train_metrics = []
        noise_std = float(TRAINING_CONFIG.get("feature_noise_std", 0.0))

        # Batch iteration
        for batch in loaders["train"]:
            batch = batch_to_device(batch)
            # Add optional jitter noise to features
            if noise_std > 0.0:
                batch = dict(batch)
                batch["features"] = batch["features"] + torch.randn_like(batch["features"]) * noise_std

            # Reset optimizer gradients, forward pass, backpropagation, and weights step
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = sallino_loss(model, batch) if method == "SALLINO" else ratio_loss(method, model, batch)
            loss.backward()
            if TRAINING_CONFIG["gradient_clip"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip"])
            optimizer.step()
            train_metrics.append(metrics)

        # Batch validation
        model.eval()
        val_metrics = []
        for batch in loaders["validation"]:
            batch = batch_to_device(batch)
            loss, metrics = sallino_loss(model, batch) if method == "SALLINO" else ratio_loss(method, model, batch)
            val_metrics.append(metrics)

        # Compute average metrics across splits
        row = {"method": method, "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        for prefix, metrics in [("train", train_metrics), ("validation", val_metrics)]:
            frame = pd.DataFrame(metrics)
            for column in frame:
                row[f"{prefix}_{column}"] = float(frame[column].mean())
        history.append(row)

        # Check early stopping progress
        val_loss = row["validation_loss"]
        if val_loss < best_val - TRAINING_CONFIG["min_delta"]:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(f"{method} epoch {epoch:03d}: lr={row['learning_rate']:.3e}, train={row['train_loss']:.4g}, val={val_loss:.4g}")
        scheduler.step()
        if stale_epochs >= TRAINING_CONFIG["patience"]:
            print(f"{method}: early stopping after {epoch} epochs; best validation loss {best_val:.4g}")
            break

    # Restore parameters of the best model checkpoint and serialize results
    if best_state is not None:
        model.load_state_dict(best_state)
    history = pd.DataFrame(history)
    torch.save(
        {
            "method": method,
            "state_dict": model.state_dict(),
            "training_config": TRAINING_CONFIG,
            "scalers": {
                "feature": {"mean": feature_scaler.mean, "scale": feature_scaler.scale},
                "theta": {"mean": theta_scaler.mean, "scale": theta_scaler.scale},
                "log_r": {"mean": log_r_scaler.mean, "scale": log_r_scaler.scale},
                "score": {"mean": score_scaler.mean, "scale": score_scaler.scale},
            },
        },
        MODEL_DIR / f"{method.lower()}.pt",
    )
    history.to_csv(MODEL_DIR / f"{method.lower()}_history.csv", index=False)
    return model, history


# ## 7. Train All Methods

# In[40]:


models = {}
histories = {}

for method in METHODS:
    model, history = train_method(method)
    models[method] = model
    histories[method] = history

print("Trained methods:", list(models))


# ## 8. Test Predictions

# In[41]:


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, guarding against constant arrays."""
    # Ensure variables are numpy arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Check that arrays are not constant to avoid division-by-zero
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def collect_ratio_predictions(method: str, model: nn.Module, loader: DataLoader) -> pd.DataFrame:
    """Collect log-ratio and score predictions on ratio test samples."""
    rows = []
    model.eval()
    # Iterate over test loader batches
    for batch in loader:
        batch = batch_to_device(batch)
        # Predict physical values
        log_r_pred, score_pred = ratio_score_from_gradient(model, batch["features"], batch["theta"])
        payload = {
            "method": method,
            "log_r_true": batch["log_r"].detach().cpu().numpy().ravel(),
            "log_r_pred": log_r_pred.detach().cpu().numpy().ravel(),
        }
        # Add ground truth and predicted scores for each operator parameter
        for i, name in enumerate(EFT_OPERATORS):
            payload[f"score_true_{name}"] = batch["score"][:, i].detach().cpu().numpy()
            payload[f"score_pred_{name}"] = score_pred[:, i].detach().cpu().numpy()
        rows.append(pd.DataFrame(payload))
    # Combine predictions to a single dataframe
    return pd.concat(rows, ignore_index=True)


def collect_sallino_predictions(model: nn.Module, loader: DataLoader) -> pd.DataFrame:
    """Collect score predictions on local-score test samples."""
    rows = []
    model.eval()
    # Iterate over local test loader batches
    for batch in loader:
        batch = batch_to_device(batch)
        # Direct score prediction
        score_pred = model(batch["features"], batch["theta"])
        payload = {"method": "SALLINO"}
        # Append predicted operator scores
        for i, name in enumerate(EFT_OPERATORS):
            payload[f"score_true_{name}"] = batch["score"][:, i].detach().cpu().numpy()
            payload[f"score_pred_{name}"] = score_pred[:, i].detach().cpu().numpy()
        rows.append(pd.DataFrame(payload))
    # Concatenate results into a single pandas dataframe
    return pd.concat(rows, ignore_index=True)


def regenerate_predictions() -> pd.DataFrame:
    """Run inference with the currently loaded models and save test predictions."""
    if "models" not in globals():
        raise NameError("models is not defined; run the training cell before regenerating predictions.")
    prediction_frames = []
    for method in ["RASCAL", "CASCAL", "ALICES"]:
        prediction_frames.append(collect_ratio_predictions(method, models[method], ratio_loaders["test"]))
    #prediction_frames.append(collect_sallino_predictions(models["SALLINO"], local_loaders["test"]))
    regenerated = pd.concat(prediction_frames, ignore_index=True)
    regenerated.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    globals()["predictions"] = regenerated
    print(f"Regenerated predictions and saved {MODEL_DIR / 'test_predictions.csv'}")
    return regenerated


def ensure_predictions() -> pd.DataFrame:
    """Return predictions from memory, current models, or the saved prediction CSV."""
    if "predictions" in globals():
        return globals()["predictions"]
    if "models" in globals():
        return regenerate_predictions()
    prediction_path = MODEL_DIR / "test_predictions.csv"
    if prediction_path.exists():
        loaded_predictions = pd.read_csv(prediction_path)
        globals()["predictions"] = loaded_predictions
        print(f"Loaded predictions from {prediction_path}")
        return loaded_predictions
    raise NameError("predictions is not defined, no saved test_predictions.csv exists, and models is not available to regenerate predictions.")


# ## 9. Metrics

# In[42]:


predictions = ensure_predictions()

metric_rows = []
for method, frame in predictions.groupby("method"):
    if "log_r_true" in frame.columns and frame["log_r_true"].notna().any():
        mask = frame["log_r_true"].notna() & frame["log_r_pred"].notna()
        residual = frame.loc[mask, "log_r_pred"] - frame.loc[mask, "log_r_true"]
        metric_rows.append({
            "method": method,
            "target": "log_r",
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "corr": pearson_corr(frame.loc[mask, "log_r_true"], frame.loc[mask, "log_r_pred"]),
        })
    for name in EFT_OPERATORS:
        true_col = f"score_true_{name}"
        pred_col = f"score_pred_{name}"
        mask = frame[true_col].notna() & frame[pred_col].notna()
        residual = frame.loc[mask, pred_col] - frame.loc[mask, true_col]
        metric_rows.append({
            "method": method,
            "target": f"score_{name}",
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "corr": pearson_corr(frame.loc[mask, true_col], frame.loc[mask, pred_col]),
        })

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(MODEL_DIR / "test_metrics.csv", index=False)
display(metrics_df.pivot(index="method", columns="target", values=["rmse", "mae", "corr"]))


# ## 10. Performance Plots

# In[43]:


def load_histories_from_disk() -> Dict[str, pd.DataFrame]:
    """Load saved training histories when the in-memory histories object is absent."""
    loaded = {}
    # Load history CSVs for each technique
    for method in METHODS:
        path = MODEL_DIR / f"{method.lower()}_history.csv"
        if path.exists():
            loaded[method] = pd.read_csv(path)
    return loaded


def ensure_predictions() -> pd.DataFrame:
    """Return test predictions from memory, disk, or currently loaded models."""
    # Check if predictions are already resident in memory
    if "predictions" in globals():
        return globals()["predictions"]

    # Prefer current in-memory models after a fresh training run.
    if "models" not in globals():
        prediction_path = MODEL_DIR / "test_predictions.csv"
        if prediction_path.exists():
            loaded_predictions = pd.read_csv(prediction_path)
            globals()["predictions"] = loaded_predictions
            print(f"Loaded predictions from {prediction_path}")
            return loaded_predictions
        raise NameError("predictions is not defined, no saved test_predictions.csv exists, and models is not available to regenerate predictions.")

    # Re-run inference across all methods to generate predictions
    prediction_path = MODEL_DIR / "test_predictions.csv"
    prediction_frames = []
    for method in ["RASCAL", "CASCAL", "ALICES"]:
        prediction_frames.append(collect_ratio_predictions(method, models[method], ratio_loaders["test"]))
    if "SALLINO" in models:
        prediction_frames.append(collect_sallino_predictions(models["SALLINO"], local_loaders["test"]))
    regenerated = pd.concat(prediction_frames, ignore_index=True)
    # Save cache prediction CSV to disk
    regenerated.to_csv(prediction_path, index=False)
    globals()["predictions"] = regenerated
    print(f"Regenerated predictions and saved {prediction_path}")
    return regenerated


def plot_history(histories: Dict[str, pd.DataFrame]) -> None:
    """Plot normalized train and validation losses with one color per method."""
    if not histories:
        print("No training histories found in memory or on disk; skipping history plot.")
        return
    # Match colors with default matplotlib color cycle
    colors = dict(zip(METHODS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    plt.figure()

    # Plot train (solid) and validation (dashed) curves scaled by initial value
    for method, history in histories.items():
        color = colors[method]
        train = history["train_loss"] / history["train_loss"].iloc[0]
        validation = history["validation_loss"] / history["train_loss"].iloc[0]
        plt.plot(history["epoch"], train, color=color, linestyle="-", label=f"{method} train")
        plt.plot(history["epoch"], validation, color=color, linestyle="--", label=f"{method} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / initial loss")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_histories.png", dpi=160)
    plt.show()


def plot_prediction_scatter(frame: pd.DataFrame, method: str, true_col: str, pred_col: str, label: str) -> None:
    """Plot predicted versus true target values as a scatter plot."""
    # Drop infinite or missing value pairs
    values = frame[[true_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return

    # Sample down points to prevent slow plotting/huge sizes
    max_points = int(TRAINING_CONFIG.get("plot_max_points", len(values)))
    if len(values) > max_points:
        values = values.sample(max_points, random_state=TRAINING_CONFIG["seed"])

    # Estimate robust plotting limits via percentiles
    low = float(np.percentile(values.to_numpy().ravel(), 1))
    high = float(np.percentile(values.to_numpy().ravel(), 99))
    if np.isclose(low, high):
        low, high = float(values.min().min()), float(values.max().max())

    # Create matplotlib scatter with a reference line
    plt.figure()
    plt.scatter(values[true_col], values[pred_col], s=8, alpha=0.35, linewidths=0)
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    #plt.xlim(-5, 5)
    #plt.ylim(-5, 5)
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(method)
    plt.tight_layout()

    # Save the output scatter plot configuration to disk
    safe_label = label.replace(" ", "_").replace("/", "_")
    plt.savefig(PLOT_DIR / f"{method.lower()}_{safe_label}_scatter.png", dpi=160)
    plt.show()


def plot_residual_histograms(prediction_table: pd.DataFrame) -> None:
    """Plot residual histograms for log-ratio and score targets."""
    # Plot log-ratio residual histograms for ratio methods
    for target, true_col, pred_col in [("log_r", "log_r_true", "log_r_pred")]:
        plt.figure()
        for method in ["RASCAL", "CASCAL", "ALICES"]:
            frame = prediction_table[prediction_table["method"] == method]
            residual = (frame[pred_col] - frame[true_col]).replace([np.inf, -np.inf], np.nan).dropna()
            plt.hist(residual, bins=80, histtype="step", density=True, label=method)
        plt.xlabel(f"Predicted - true {target}")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{target}_residuals.png", dpi=160)
        plt.show()

    # Plot score residual histograms for all estimators
    for name in EFT_OPERATORS:
        plt.figure()
        true_col = f"score_true_{name}"
        pred_col = f"score_pred_{name}"
        for method in METHODS:
            frame = prediction_table[prediction_table["method"] == method]
            residual = (frame[pred_col] - frame[true_col]).replace([np.inf, -np.inf], np.nan).dropna()
            plt.hist(residual, bins=80, histtype="step", density=True, label=method)
        plt.xlabel(f"Predicted - true score {name}")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"score_{name}_residuals.png", dpi=160)
        plt.show()


plot_histories = globals().get("histories", load_histories_from_disk())
predictions = ensure_predictions()

plot_history(plot_histories)

for method in ["RASCAL", "CASCAL", "ALICES"]:
    frame = predictions[predictions["method"] == method]
    plot_prediction_scatter(frame, method, "log_r_true", "log_r_pred", "log_r")
    for name in EFT_OPERATORS:
        plot_prediction_scatter(frame, method, f"score_true_{name}", f"score_pred_{name}", f"score_{name}")

if "SALLINO" in set(predictions["method"]):
    frame = predictions[predictions["method"] == "SALLINO"]
    for name in EFT_OPERATORS:
        plot_prediction_scatter(frame, "SALLINO", f"score_true_{name}", f"score_pred_{name}", f"score_{name}")

plot_residual_histograms(predictions)


# In[ ]:
