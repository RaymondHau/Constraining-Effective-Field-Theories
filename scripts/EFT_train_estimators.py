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
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

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
SCRIPT_START = time.monotonic()
LAST_TIMING_MARK = SCRIPT_START


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as compact human-readable time."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {remainder:.1f}s"


def log_timing(label: str) -> None:
    """Print time spent since the previous checkpoint and since script start."""
    global LAST_TIMING_MARK
    now = time.monotonic()
    print(
        f"[timing] {label}: step={format_duration(now - LAST_TIMING_MARK)}, "
        f"total={format_duration(now - SCRIPT_START)}",
        flush=True,
    )
    LAST_TIMING_MARK = now


def progress_text(done: int, total: int | None) -> str:
    """Return row progress with percentage when a total is available."""
    if total and total > 0:
        return f"{done:,}/{total:,} rows ({100.0 * done / total:.1f}%)"
    return f"{done:,} rows"


def progress_percent_line(done: int, total: int | None, elapsed: float) -> str:
    """Return compact percent progress and percent-per-second throughput."""
    if total and total > 0:
        percent = 100.0 * done / total
        percent_rate = percent / elapsed if elapsed > 0.0 else float("nan")
        return f"[progress] ({percent:.1f}%) in {format_duration(elapsed)} ({percent_rate:.2f} percent per second)"
    rate = done / elapsed if elapsed > 0.0 else float("nan")
    return f"[progress] ({done:,} rows) in {format_duration(elapsed)} ({rate:,.0f} rows per second)"


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
METHOD_CONFIGS = {method: {} for method in METHODS}
RATIO_METHODS = {"RASCAL", "CASCAL", "ALICES"}

TRAINING_CONFIG = {
    "batch_size": 4096,
    "epochs": 200,
    "learning_rate": 0.0015,
    "min_learning_rate": 1.0e-5,
    "weight_decay": 0, # 0.25,
    "hidden_layers": [1024, 1024, 1024, 1024, 1024],    #[1024, 1024, 256, 128]
    "dropout": 0.0,
    "feature_noise_std": 0.0,
    "gradient_clip": 1000000.0,
    "patience": 200,
    "min_delta": 1.0e-6,
    "seed": 1234,
    "activation": "tanh",
    "csv_chunk_rows": 100_000,
    "cache_train_tensors": True,
    "training_cache_subdir": "tensor_cache",
    "group_ratio_methods": True,
}


def normalize_methods(raw_methods: Iterable[object]) -> Tuple[List[str], Dict[str, Dict[str, object]]]:
    """Return method names and per-method config from JSON objects."""
    names: List[str] = []
    configs: Dict[str, Dict[str, object]] = {}
    for entry in raw_methods:
        if not isinstance(entry, Mapping):
            raise TypeError(f"Unsupported method entry {entry!r}; expected an object with name and alpha.")
        name = str(entry.get("name", entry.get("method"))).upper()
        if name in {"", "NONE"}:
            raise ValueError(f"Method config entry is missing a name: {entry!r}")
        method_config = {str(key): value for key, value in entry.items() if key not in {"name", "method"}}
        if "alpha" not in method_config:
            raise KeyError(f"Method {name!r} is missing required alpha.")
        names.append(name)
        configs[name] = method_config
    return names, configs


def method_alpha(method: str) -> float:
    """Return the score-loss weight configured for one estimator."""
    if "alpha" not in METHOD_CONFIGS.get(method, {}):
        raise KeyError(f"Missing required alpha for training method {method!r}.")
    alpha = float(METHOD_CONFIGS[method]["alpha"])
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"Alpha for {method!r} must be finite and non-negative, got {alpha!r}.")
    return alpha


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
SCORE_COORDINATE_SCALE = np.array(
    [_PHYSICS_CONFIG.get("morphing_theta_scale", {}).get(name, 1.0) for name in EFT_OPERATORS],
    dtype=np.float32,
).reshape(1, -1)
if np.any(~np.isfinite(SCORE_COORDINATE_SCALE)) or np.any(SCORE_COORDINATE_SCALE <= 0.0):
    raise ValueError(f"morphing_theta_scale must contain positive finite values, got {SCORE_COORDINATE_SCALE!r}")
if "training_config" in _TRAINING_SECTION:
    TRAINING_CONFIG.update(_TRAINING_SECTION["training_config"])
if "training_config" in _STAGE_CONFIG:
    TRAINING_CONFIG.update(_STAGE_CONFIG["training_config"])
if "methods" in _TRAINING_SECTION:
    METHODS, METHOD_CONFIGS = normalize_methods(_TRAINING_SECTION["methods"])
if "methods" in _STAGE_CONFIG:
    METHODS, METHOD_CONFIGS = normalize_methods(_STAGE_CONFIG["methods"])
CONFIGURED_RATIO_METHODS = [method for method in METHODS if method in RATIO_METHODS]
print("Training methods:", METHODS)
print("Method configs:", METHOD_CONFIGS)
print("Training config:", TRAINING_CONFIG)
print("Score coordinate scale:", dict(zip(EFT_OPERATORS, SCORE_COORDINATE_SCALE.ravel())))
log_timing("configuration loaded")


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
    if split == "train" or "SALLINO" in METHODS:
        validate_required_columns(local_paths[split], required_local)

ratio_frames = {split: read_required_csv(ratio_paths[split], usecols=required_ratio) for split in ["validation", "test"]}
local_frames = {}
if "SALLINO" in METHODS:
    local_frames = {split: read_required_csv(local_paths[split], usecols=required_local) for split in ["validation", "test"]}

summary_path = INPUT_DIR / "sample_summary.csv"
if summary_path.exists():
    sample_summary = pd.read_csv(summary_path)
    print("Prepared sample rows:")
    display(sample_summary)
else:
    print("Prepared sample row summary not found; train CSVs will be streamed without pre-counting.")
print("Loaded validation/test ratio rows:", {split: len(frame) for split, frame in ratio_frames.items()})
if local_frames:
    print("Loaded validation/test local rows:", {split: len(frame) for split, frame in local_frames.items()})
log_timing("validated headers and loaded validation/test samples")


def expected_rows(kind: str, split: str) -> int | None:
    """Return expected rows from sample_summary.csv when available."""
    if "sample_summary" not in globals():
        return None
    row = sample_summary.loc[sample_summary["split"] == split]
    column = f"{kind}_rows"
    if row.empty or column not in row.columns:
        return None
    value = row.iloc[0][column]
    if pd.isna(value):
        return None
    return int(value)


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


def empty_stats(width: int) -> Dict[str, object]:
    return {"count": 0, "sum": np.zeros(width, dtype=np.float64), "sumsq": np.zeros(width, dtype=np.float64)}


def update_stats(stats: Dict[str, object], values: np.ndarray) -> None:
    stats["count"] = int(stats["count"]) + len(values)
    stats["sum"] += values.sum(axis=0)
    stats["sumsq"] += np.square(values).sum(axis=0)


def stats_to_standardizer(stats: Dict[str, object]) -> Standardizer:
    count = int(stats["count"])
    if count == 0:
        raise RuntimeError("Cannot fit scaler from empty CSV input")
    mean_1d = stats["sum"] / count
    variance_1d = np.maximum(stats["sumsq"] / count - np.square(mean_1d), 0.0)
    scale_1d = np.sqrt(variance_1d)
    scale_1d = np.where(scale_1d > 0.0, scale_1d, 1.0)
    return Standardizer(mean_1d[None, :].astype(np.float32), scale_1d[None, :].astype(np.float32))


def fit_training_scalers() -> Tuple[Standardizer, Standardizer, Standardizer, Standardizer, Standardizer]:
    """Fit each input and target scaler only from its corresponding training sample."""
    feature_stats = empty_stats(len(FEATURE_COLUMNS))
    theta_stats = empty_stats(len(theta0_columns))
    log_r_stats = empty_stats(1)
    ratio_score_stats = empty_stats(len(score_columns))

    ratio_columns = list(dict.fromkeys(FEATURE_COLUMNS + theta0_columns + ["y", "log_r"] + score_columns))
    print(f"Fitting ratio scalers from {ratio_paths['train'].name}", flush=True)
    ratio_count = 0
    ratio_start = time.monotonic()
    ratio_total = expected_rows("ratio", "train")
    for chunk_index, chunk in enumerate(pd.read_csv(ratio_paths["train"], usecols=ratio_columns, chunksize=CSV_CHUNK_ROWS), start=1):
        update_stats(feature_stats, chunk[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
        update_stats(theta_stats, chunk[theta0_columns].to_numpy(dtype=np.float64))
        update_stats(log_r_stats, chunk[["log_r"]].to_numpy(dtype=np.float64))
        numerator_chunk = chunk.loc[chunk["y"] == 1.0, score_columns]
        update_stats(ratio_score_stats, numerator_chunk.to_numpy(dtype=np.float64))
        ratio_count += len(chunk)
        if chunk_index == 1 or chunk_index % 10 == 0:
            elapsed = time.monotonic() - ratio_start
            rate = ratio_count / elapsed if elapsed > 0.0 else float("nan")
            print(
                f"  {ratio_paths['train'].name}: scaler progress {progress_text(ratio_count, ratio_total)} "
                f"in {format_duration(elapsed)} ({rate:,.0f} rows/s)",
                flush=True,
            )

    ratio_score_scaler = stats_to_standardizer(ratio_score_stats)
    local_score_scaler = ratio_score_scaler
    if "SALLINO" in METHODS:
        local_score_stats = empty_stats(len(score_columns))
        print(f"Fitting local score scaler from {local_paths['train'].name}", flush=True)
        for chunk in pd.read_csv(local_paths["train"], usecols=score_columns, chunksize=CSV_CHUNK_ROWS):
            update_stats(local_score_stats, chunk[score_columns].to_numpy(dtype=np.float64))
        local_score_scaler = stats_to_standardizer(local_score_stats)

    return (
        stats_to_standardizer(feature_stats),
        stats_to_standardizer(theta_stats),
        stats_to_standardizer(log_r_stats),
        ratio_score_scaler,
        local_score_scaler,
    )


feature_scaler, theta_scaler, log_r_scaler, ratio_score_scaler, local_score_scaler = fit_training_scalers()
log_timing("fitted training scalers")


def scaler_signature(scaler: Standardizer) -> Dict[str, object]:
    """Return JSON-serializable scaler metadata for cache invalidation."""
    return {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()}


def cache_signature(kind: str, path: Path, usecols: list[str]) -> Tuple[str, Dict[str, object]]:
    """Return a stable cache key and manifest payload for preprocessed tensors."""
    stat = path.stat()
    payload = {
        "schema_version": 2,
        "kind": kind,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "usecols": usecols,
        "feature_columns": FEATURE_COLUMNS,
        "theta0_columns": theta0_columns,
        "theta_columns": theta_columns,
        "score_columns": score_columns,
        "score_coordinate_scale": SCORE_COORDINATE_SCALE.tolist(),
        "csv_chunk_rows": CSV_CHUNK_ROWS,
        "scalers": {
            "feature": scaler_signature(feature_scaler),
            "theta": scaler_signature(theta_scaler),
            "log_r": scaler_signature(log_r_scaler),
            "score": scaler_signature(local_score_scaler if kind == "local" else ratio_score_scaler),
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], payload


def torch_load_cpu(path: Path) -> Dict[str, torch.Tensor]:
    """Load a tensor chunk on CPU across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_ratio_payload(frame: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Convert one ratio CSV chunk into standardized CPU tensors."""
    return {
        "features": torch.as_tensor(feature_scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)), dtype=torch.float32),
        "theta": torch.as_tensor(theta_scaler.transform(frame[theta0_columns].to_numpy(dtype=np.float32)), dtype=torch.float32),
        "y": torch.as_tensor(frame[["y"]].to_numpy(dtype=np.float32), dtype=torch.float32),
        "soft_y": torch.as_tensor(frame[["soft_y"]].to_numpy(dtype=np.float32), dtype=torch.float32),
        "log_r": torch.as_tensor(frame[["log_r"]].to_numpy(dtype=np.float32), dtype=torch.float32),
        "likelihood_ratio": torch.as_tensor(frame[["likelihood_ratio"]].to_numpy(dtype=np.float32), dtype=torch.float32),
        "score": torch.as_tensor(frame[score_columns].to_numpy(dtype=np.float32), dtype=torch.float32),
    }


def build_local_payload(frame: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Convert one local-score CSV chunk into standardized CPU tensors."""
    return {
        "features": torch.as_tensor(feature_scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)), dtype=torch.float32),
        "theta": torch.as_tensor(theta_scaler.transform(frame[theta_columns].to_numpy(dtype=np.float32)), dtype=torch.float32),
        "score": torch.as_tensor(frame[score_columns].to_numpy(dtype=np.float32), dtype=torch.float32),
    }


def prepare_tensor_cache(
    kind: str,
    path: Path,
    usecols: list[str],
    payload_builder,
) -> list[Path] | None:
    """Build or reuse standardized training tensor chunks for a CSV source."""
    if not bool(TRAINING_CONFIG.get("cache_train_tensors", True)):
        return None

    key, manifest_payload = cache_signature(kind, path, usecols)
    cache_root = INPUT_DIR / str(TRAINING_CONFIG.get("training_cache_subdir", "tensor_cache"))
    cache_dir = cache_root / f"{kind}_{key}"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        chunk_paths = [cache_dir / name for name in manifest.get("chunks", [])]
        if manifest.get("signature") == manifest_payload and chunk_paths and all(chunk.exists() for chunk in chunk_paths):
            print(f"Using cached {kind} training tensors from {cache_dir}", flush=True)
            return chunk_paths

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building cached {kind} training tensors in {cache_dir}", flush=True)
    chunk_names = []
    row_count = 0
    cache_start = time.monotonic()
    expected_total = expected_rows(kind, "train")
    for chunk_index, frame in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CSV_CHUNK_ROWS), start=1):
        payload = payload_builder(frame)
        chunk_name = f"chunk_{chunk_index:06d}.pt"
        torch.save(payload, cache_dir / chunk_name)
        chunk_names.append(chunk_name)
        row_count += len(frame)
        if chunk_index == 1 or chunk_index % 10 == 0:
            elapsed = time.monotonic() - cache_start
            rate = row_count / elapsed if elapsed > 0.0 else float("nan")
            print(
                f"  cached {kind} chunk {chunk_index}: {progress_text(row_count, expected_total)} "
                f"in {format_duration(elapsed)} ({rate:,.0f} rows/s)",
                flush=True,
            )

    manifest = {"signature": manifest_payload, "chunks": chunk_names, "rows": row_count}
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return [cache_dir / name for name in chunk_names]


def yield_payload_batches(payload: Dict[str, torch.Tensor], batch_size: int):
    """Yield fixed-size batch slices from one tensor payload."""
    rows = int(payload["features"].shape[0])
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        yield {key: value[start:stop] for key, value in payload.items()}


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
        self.likelihood_ratio = torch.as_tensor(frame[["likelihood_ratio"]].to_numpy(dtype=np.float32), dtype=torch.float32)

        self.score = torch.as_tensor(frame[score_columns].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one ratio-training row."""
        # Package and return a single event batch item
        keys = ["features", "theta", "y", "soft_y", "log_r", "likelihood_ratio", "score"]
        return {key: getattr(self, key)[index] for key in keys}


class LocalScoreDataset(Dataset):
    """Dataset for SALLINO direct-score training."""
    def __init__(self, frame: pd.DataFrame):
        """Convert a local-score sample DataFrame to tensors."""
        # Scale features and local physical parameters
        self.features = torch.as_tensor(feature_scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)), dtype=torch.float32)
        self.theta = torch.as_tensor(theta_scaler.transform(frame[theta_columns].to_numpy(dtype=np.float32)), dtype=torch.float32)

        # Load true local score components
        self.score = torch.as_tensor(frame[score_columns].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one local-score row."""
        # Fetch properties as dictionary values
        return {key: getattr(self, key)[index] for key in ["features", "theta", "score"]}


class RatioCSVBatchDataset(IterableDataset):
    """Stream ratio training batches from CSV chunks."""
    def __init__(self, path: Path):
        self.path = path
        self.cache_chunks = prepare_tensor_cache("ratio", path, required_ratio, build_ratio_payload)

    def __iter__(self):
        batch_size = int(TRAINING_CONFIG["batch_size"])
        if self.cache_chunks is not None:
            for chunk_index, chunk_path in enumerate(self.cache_chunks, start=1):
                if chunk_index == 1 or chunk_index % 10 == 0:
                    print(f"Streaming cached {self.path.name} chunk {chunk_index}", flush=True)
                yield from yield_payload_batches(torch_load_cpu(chunk_path), batch_size)
            return

        for chunk_index, frame in enumerate(pd.read_csv(self.path, usecols=required_ratio, chunksize=CSV_CHUNK_ROWS), start=1):
            if chunk_index == 1 or chunk_index % 10 == 0:
                print(f"Streaming {self.path.name} chunk {chunk_index} ({len(frame):,} rows)", flush=True)
            yield from yield_payload_batches(build_ratio_payload(frame), batch_size)


class LocalCSVBatchDataset(IterableDataset):
    """Stream local-score training batches from CSV chunks."""
    def __init__(self, path: Path):
        self.path = path
        self.cache_chunks = prepare_tensor_cache("local", path, required_local, build_local_payload)

    def __iter__(self):
        batch_size = int(TRAINING_CONFIG["batch_size"])
        if self.cache_chunks is not None:
            for chunk_index, chunk_path in enumerate(self.cache_chunks, start=1):
                if chunk_index == 1 or chunk_index % 10 == 0:
                    print(f"Streaming cached {self.path.name} chunk {chunk_index}", flush=True)
                yield from yield_payload_batches(torch_load_cpu(chunk_path), batch_size)
            return

        for chunk_index, frame in enumerate(pd.read_csv(self.path, usecols=required_local, chunksize=CSV_CHUNK_ROWS), start=1):
            if chunk_index == 1 or chunk_index % 10 == 0:
                print(f"Streaming {self.path.name} chunk {chunk_index} ({len(frame):,} rows)", flush=True)
            yield from yield_payload_batches(build_local_payload(frame), batch_size)


def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    """Return a DataLoader with the configured batch size."""
    # Iterable CSV datasets yield already-batched dictionaries.
    if isinstance(dataset, IterableDataset):
        return DataLoader(dataset, batch_size=None, pin_memory=(DEVICE == "cuda"))
    return DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(DEVICE == "cuda"),
    )


ratio_loaders = {
    "train": make_loader(RatioCSVBatchDataset(ratio_paths["train"]), shuffle=False),
    **{split: make_loader(RatioDataset(frame), shuffle=False) for split, frame in ratio_frames.items()},
}
local_loaders = {}
if "SALLINO" in METHODS:
    local_loaders = {
        "train": make_loader(LocalCSVBatchDataset(local_paths["train"]), shuffle=False),
        **{split: make_loader(LocalScoreDataset(frame), shuffle=False) for split, frame in local_frames.items()},
    }
log_timing("built datasets, loaders, and training tensor caches")


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
    return {key: value.to(DEVICE, non_blocking=(DEVICE == "cuda")) for key, value in batch.items()}


def set_parameter_grad_state(model: nn.Module, requires_grad: bool) -> list[bool]:
    """Set parameter grad tracking and return the previous states."""
    previous = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad)
    return previous


def restore_parameter_grad_state(model: nn.Module, previous: list[bool]) -> None:
    """Restore parameter grad tracking after validation or inference."""
    for parameter, requires_grad in zip(model.parameters(), previous):
        parameter.requires_grad_(requires_grad)


def normalized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: np.ndarray,
) -> torch.Tensor:
    """Return MSE of residuals normalized by each target's training std."""
    scale_tensor = torch.as_tensor(scale, dtype=prediction.dtype, device=prediction.device)
    return torch.mean(torch.square((prediction - target) / scale_tensor))


def numerator_score_loss(
    score_prediction: torch.Tensor,
    score_target: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return N^-1 sum of score errors on numerator (y=1) events only."""
    per_event = torch.sum(torch.square(score_prediction - score_target), dim=1, keepdim=True)
    return torch.mean(y * per_event)


def score_regression_loss(score_prediction: torch.Tensor, score_target: torch.Tensor) -> torch.Tensor:
    """Return the mean per-event squared Euclidean score error."""
    return torch.mean(torch.sum(torch.square(score_prediction - score_target), dim=1))


def classifier_augmented_loss(
    log_r_prediction: torch.Tensor,
    class_target: torch.Tensor,
    score_prediction: torch.Tensor,
    score_target: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return classifier loss plus numerator-only score regression."""
    main_loss = nn.functional.binary_cross_entropy_with_logits(log_r_prediction, class_target)
    score_loss = numerator_score_loss(score_prediction, score_target, y)
    return main_loss + alpha * score_loss, main_loss, score_loss


def rascal_ratio_loss(
    log_r_prediction: torch.Tensor,
    ratio_target: torch.Tensor,
    y: torch.Tensor,
    max_abs_log_ratio: float = 30.0,
) -> torch.Tensor:
    """Return Eq. (37)'s ratio/inverse-ratio regression for y=1 numerator labels."""
    clipped_log_r = torch.clamp(log_r_prediction, -max_abs_log_ratio, max_abs_log_ratio)
    ratio_prediction = torch.exp(clipped_log_r)
    inverse_ratio_prediction = torch.exp(-clipped_log_r)
    safe_ratio_target = torch.clamp(ratio_target, min=torch.finfo(ratio_target.dtype).tiny)
    denominator_term = (1.0 - y) * torch.square(ratio_prediction - safe_ratio_target)
    numerator_term = y * torch.square(inverse_ratio_prediction - torch.reciprocal(safe_ratio_target))
    return torch.mean(denominator_term + numerator_term)


def rascal_augmented_loss(
    log_r_prediction: torch.Tensor,
    ratio_target: torch.Tensor,
    score_prediction: torch.Tensor,
    score_target: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    max_abs_log_ratio: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the RASCAL ratio loss plus numerator-only score regression."""
    main_loss = rascal_ratio_loss(log_r_prediction, ratio_target, y, max_abs_log_ratio)
    score_loss = numerator_score_loss(score_prediction, score_target, y)
    return main_loss + alpha * score_loss, main_loss, score_loss


def ratio_loss(method: str, model: RatioEstimator, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return the method-specific ratio loss for one batch."""
    # Predict log r and its gradient-based score
    log_r_pred, score_pred = ratio_score_from_gradient(model, batch["features"], batch["theta"])
    score_coordinate_scale = torch.as_tensor(
        SCORE_COORDINATE_SCALE, dtype=score_pred.dtype, device=score_pred.device
    )
    scaled_score_pred = score_pred * score_coordinate_scale
    scaled_score_target = batch["score"] * score_coordinate_scale

    alpha = method_alpha(method)
    # This workspace uses y=1 for numerator events and y=0 for denominator events.
    # The paper's score term is therefore masked by y in every augmented method.
    if method == "ALICES":
        total, main_loss, score_loss = classifier_augmented_loss(
            log_r_pred, batch["soft_y"], scaled_score_pred, scaled_score_target, batch["y"], alpha
        )
    elif method == "CASCAL":
        total, main_loss, score_loss = classifier_augmented_loss(
            log_r_pred, batch["y"], scaled_score_pred, scaled_score_target, batch["y"], alpha
        )
    elif method == "RASCAL":
        total, main_loss, score_loss = rascal_augmented_loss(
            log_r_pred,
            batch["likelihood_ratio"],
            scaled_score_pred,
            scaled_score_target,
            batch["y"],
            alpha,
            float(TRAINING_CONFIG.get("rascal_max_abs_log_ratio", 30.0)),
        )
    else:
        raise ValueError(method)

    weighted_score_loss = alpha * score_loss
    log_r_mse = torch.mean(torch.square(log_r_pred - batch["log_r"]))
    classifier_bce = nn.functional.binary_cross_entropy_with_logits(log_r_pred, batch["y"])
    return total, {
        "loss": total.detach(),
        "main_loss": main_loss.detach(),
        "score_loss": score_loss.detach(),
        "weighted_score_loss": weighted_score_loss.detach(),
        "log_r_mse": log_r_mse.detach(),
        "classifier_bce": classifier_bce.detach(),
        "alpha": alpha,
        "batch_rows": int(log_r_pred.shape[0]),
    }


def sallino_loss(model: ScoreEstimator, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return the SALLINO direct score loss."""
    # Feed features and theta, compute prediction score
    score_pred = model(batch["features"], batch["theta"])
    score_coordinate_scale = torch.as_tensor(
        SCORE_COORDINATE_SCALE, dtype=score_pred.dtype, device=score_pred.device
    )
    loss = score_regression_loss(
        score_pred * score_coordinate_scale,
        batch["score"] * score_coordinate_scale,
    )
    return loss, {
        "loss": loss.detach(),
        "main_loss": math.nan,
        "score_loss": loss.detach(),
        "batch_rows": int(score_pred.shape[0]),
    }


def mean_metric_values(metrics: list[Dict[str, object]]) -> Dict[str, float]:
    """Average batch metrics by row count with one device sync per metric."""
    if not metrics:
        return {}
    row_weights = np.asarray([row.get("batch_rows", 1) for row in metrics], dtype=np.float64)
    total_rows = float(row_weights.sum())
    averaged = {}
    for column in metrics[0]:
        if column == "batch_rows":
            continue
        values = [row[column] for row in metrics]
        tensor_values = [value for value in values if torch.is_tensor(value)]
        if tensor_values:
            stacked = torch.stack(tensor_values)
            weights = torch.as_tensor(row_weights, dtype=stacked.dtype, device=stacked.device)
            averaged[column] = float((stacked * weights).sum().div(total_rows).detach().cpu())
        else:
            numeric = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(numeric)
            averaged[column] = (
                float(np.sum(numeric[finite] * row_weights[finite]) / np.sum(row_weights[finite]))
                if np.any(finite)
                else float("nan")
            )
    return averaged


def make_method_components(method: str) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, Dict[str, DataLoader]]:
    """Create the model, optimizer, scheduler, and loaders for one method."""
    torch.manual_seed(TRAINING_CONFIG["seed"])
    input_dim = len(FEATURE_COLUMNS) + len(EFT_OPERATORS)
    if method == "SALLINO":
        model = ScoreEstimator(input_dim, len(EFT_OPERATORS), TRAINING_CONFIG["hidden_layers"]).to(DEVICE)
        loaders = local_loaders
    else:
        model = RatioEstimator(input_dim, TRAINING_CONFIG["hidden_layers"]).to(DEVICE)
        loaders = ratio_loaders

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"], weight_decay=TRAINING_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG["epochs"],
        eta_min=TRAINING_CONFIG["min_learning_rate"],
    )
    return model, optimizer, scheduler, loaders


def save_trained_method(method: str, model: nn.Module, history_rows: list[Dict[str, float]] | pd.DataFrame) -> pd.DataFrame:
    """Save a trained model checkpoint and history CSV."""
    history = history_rows if isinstance(history_rows, pd.DataFrame) else pd.DataFrame(history_rows)
    effective_training_config = dict(TRAINING_CONFIG)
    effective_training_config["alpha"] = method_alpha(method)
    effective_training_config["loss_convention"] = (
        "paper_ratio_inverse_ratio_plus_numerator_score"
        if method == "RASCAL"
        else "classifier_plus_numerator_score"
    )
    effective_training_config["score_coordinate_scale"] = dict(
        zip(EFT_OPERATORS, SCORE_COORDINATE_SCALE.ravel().tolist())
    )
    torch.save(
        {
            "method": method,
            "state_dict": model.state_dict(),
            "training_config": effective_training_config,
            "method_config": dict(METHOD_CONFIGS.get(method, {})),
            "scalers": {
                "feature": {"mean": feature_scaler.mean, "scale": feature_scaler.scale},
                "theta": {"mean": theta_scaler.mean, "scale": theta_scaler.scale},
                "log_r": {"mean": log_r_scaler.mean, "scale": log_r_scaler.scale},
                "score": {
                    "mean": (local_score_scaler if method == "SALLINO" else ratio_score_scaler).mean,
                    "scale": (local_score_scaler if method == "SALLINO" else ratio_score_scaler).scale,
                },
            },
        },
        MODEL_DIR / f"{method.lower()}.pt",
    )
    history.to_csv(MODEL_DIR / f"{method.lower()}_history.csv", index=False)
    return history


def train_method(method: str, method_index: int, method_count: int) -> Tuple[nn.Module, pd.DataFrame]:
    """Train one estimator and return the best model plus history."""
    method_start = time.monotonic()
    print(f"[progress] Starting {method} ({method_index}/{method_count})", flush=True)
    model, optimizer, scheduler, loaders = make_method_components(method)

    print(f"{method}: {sum(isinstance(module, nn.Dropout) for module in model.modules())} dropout layers, p={TRAINING_CONFIG['dropout']}")
    log_timing(f"{method} model, optimizer, and scheduler initialized")
    history = []
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    epochs = int(TRAINING_CONFIG["epochs"])
    progress_interval_rows = int(TRAINING_CONFIG.get("progress_interval_rows", max(int(TRAINING_CONFIG["batch_size"]) * 10, 1)))
    train_total = expected_rows("local" if method == "SALLINO" else "ratio", "train")
    validation_total = expected_rows("local" if method == "SALLINO" else "ratio", "validation")
    training_loop_start = time.monotonic()

    # Start training epoch loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[progress] {method} epoch {epoch}/{epochs}, methods ['{method}'], lr={current_lr:.3e}", flush=True)
        model.train()
        train_metrics = []
        noise_std = float(TRAINING_CONFIG.get("feature_noise_std", 0.0))
        train_rows = 0
        train_start = time.monotonic()
        next_train_progress = progress_interval_rows

        # Batch iteration
        for batch in loaders["train"]:
            batch = batch_to_device(batch)
            train_rows += int(batch["features"].shape[0])
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
            if train_rows >= next_train_progress or (train_total and train_rows >= train_total):
                elapsed = time.monotonic() - train_start
                print(progress_percent_line(train_rows, train_total, elapsed), flush=True)
                next_train_progress += progress_interval_rows
        train_elapsed = time.monotonic() - train_start

        # Batch validation
        model.eval()
        val_metrics = []
        validation_rows = 0
        validation_start = time.monotonic()
        parameter_grad_state = set_parameter_grad_state(model, False)
        try:
            for batch in loaders["validation"]:
                batch = batch_to_device(batch)
                validation_rows += int(batch["features"].shape[0])
                loss, metrics = sallino_loss(model, batch) if method == "SALLINO" else ratio_loss(method, model, batch)
                val_metrics.append(metrics)
        finally:
            restore_parameter_grad_state(model, parameter_grad_state)
        validation_elapsed = time.monotonic() - validation_start

        # Compute average metrics across splits
        row = {"method": method, "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        for prefix, metrics in [("train", train_metrics), ("validation", val_metrics)]:
            for column, value in mean_metric_values(metrics).items():
                row[f"{prefix}_{column}"] = value
        history.append(row)

        # Check early stopping progress
        selection_column = "validation_classifier_bce" if bool(TRAINING_CONFIG.get("optuna_scan_mode", False)) and method in RATIO_METHODS else "validation_loss"
        val_loss = row[selection_column]
        if val_loss < best_val - TRAINING_CONFIG["min_delta"]:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        epoch_elapsed = time.monotonic() - epoch_start
        print(
            f"[results] epoch {epoch:03d}, epoch time={epoch_elapsed:.1f}s, total time={time.monotonic() - training_loop_start:.1f}s:\n"
            f"{method} train={row['train_loss']:.4g}, val={val_loss:.4g}",
            flush=True,
        )
        scheduler.step()
        if stale_epochs >= TRAINING_CONFIG["patience"]:
            print(f"{method}: early stopping after {epoch} epochs; best validation loss {best_val:.4g}")
            break

    # Restore parameters of the best model checkpoint and serialize results
    if best_state is not None:
        model.load_state_dict(best_state)
    history = save_trained_method(method, model, history)
    print(f"[timing] {method} total training and save time: {format_duration(time.monotonic() - method_start)}", flush=True)
    return model, history


def train_ratio_methods_grouped(ratio_methods: list[str], start_index: int, method_count: int) -> Tuple[Dict[str, nn.Module], Dict[str, pd.DataFrame]]:
    """Train ratio estimators together, reusing each streamed train/validation batch."""
    group_start = time.monotonic()
    print(
        f"[progress] Starting grouped ratio training for {ratio_methods} "
        f"(methods {start_index}-{start_index + len(ratio_methods) - 1}/{method_count})",
        flush=True,
    )

    models: Dict[str, nn.Module] = {}
    optimizers = {}
    schedulers = {}
    histories: Dict[str, list[Dict[str, float]]] = {method: [] for method in ratio_methods}
    best_states = {method: None for method in ratio_methods}
    best_values = {method: float("inf") for method in ratio_methods}
    stale_epochs = {method: 0 for method in ratio_methods}
    active = {method: True for method in ratio_methods}

    for method in ratio_methods:
        model, optimizer, scheduler, _ = make_method_components(method)
        models[method] = model
        optimizers[method] = optimizer
        schedulers[method] = scheduler
        print(f"{method}: {sum(isinstance(module, nn.Dropout) for module in model.modules())} dropout layers, p={TRAINING_CONFIG['dropout']}")
    log_timing("grouped ratio models, optimizers, and schedulers initialized")

    epochs = int(TRAINING_CONFIG["epochs"])
    progress_interval_rows = int(TRAINING_CONFIG.get("progress_interval_rows", max(int(TRAINING_CONFIG["batch_size"]) * 10, 1)))
    train_total = expected_rows("ratio", "train")
    validation_total = expected_rows("ratio", "validation")
    noise_std = float(TRAINING_CONFIG.get("feature_noise_std", 0.0))
    training_loop_start = time.monotonic()

    for epoch in range(1, epochs + 1):
        active_methods = [method for method in ratio_methods if active[method]]
        if not active_methods:
            print("[progress] All grouped ratio methods stopped early.", flush=True)
            break

        epoch_start = time.monotonic()
        current_lr = optimizers[active_methods[0]].param_groups[0]["lr"]
        print(f"[progress] grouped ratio epoch {epoch}/{epochs}, methods {active_methods}, lr={current_lr:.3e}", flush=True)
        for method in active_methods:
            models[method].train()

        train_metrics = {method: [] for method in active_methods}
        train_rows = 0
        train_start = time.monotonic()
        next_train_progress = progress_interval_rows

        for batch in ratio_loaders["train"]:
            batch = batch_to_device(batch)
            train_rows += int(batch["features"].shape[0])
            for method in active_methods:
                method_batch = batch
                if noise_std > 0.0:
                    method_batch = dict(batch)
                    method_batch["features"] = batch["features"] + torch.randn_like(batch["features"]) * noise_std

                optimizers[method].zero_grad(set_to_none=True)
                loss, metrics = ratio_loss(method, models[method], method_batch)
                loss.backward()
                if TRAINING_CONFIG["gradient_clip"] is not None:
                    nn.utils.clip_grad_norm_(models[method].parameters(), TRAINING_CONFIG["gradient_clip"])
                optimizers[method].step()
                train_metrics[method].append(metrics)

            if train_rows >= next_train_progress or (train_total and train_rows >= train_total):
                elapsed = time.monotonic() - train_start
                print(progress_percent_line(train_rows, train_total, elapsed), flush=True)
                next_train_progress += progress_interval_rows
        train_elapsed = time.monotonic() - train_start

        for method in active_methods:
            models[method].eval()
        val_metrics = {method: [] for method in active_methods}
        validation_rows = 0
        validation_start = time.monotonic()
        parameter_grad_states = {method: set_parameter_grad_state(models[method], False) for method in active_methods}
        try:
            for batch in ratio_loaders["validation"]:
                batch = batch_to_device(batch)
                validation_rows += int(batch["features"].shape[0])
                for method in active_methods:
                    loss, metrics = ratio_loss(method, models[method], batch)
                    val_metrics[method].append(metrics)
        finally:
            for method in active_methods:
                restore_parameter_grad_state(models[method], parameter_grad_states[method])
        validation_elapsed = time.monotonic() - validation_start

        epoch_elapsed = time.monotonic() - epoch_start
        result_lines = [
            f"[results] epoch {epoch:03d}, epoch time={epoch_elapsed:.1f}s, total time={time.monotonic() - training_loop_start:.1f}s:"
        ]
        for method in active_methods:
            row = {"method": method, "epoch": epoch, "learning_rate": optimizers[method].param_groups[0]["lr"]}
            for prefix, metrics in [("train", train_metrics[method]), ("validation", val_metrics[method])]:
                for column, value in mean_metric_values(metrics).items():
                    row[f"{prefix}_{column}"] = value
            histories[method].append(row)

            selection_column = "validation_classifier_bce" if bool(TRAINING_CONFIG.get("optuna_scan_mode", False)) else "validation_loss"
            val_loss = row[selection_column]
            if val_loss < best_values[method] - TRAINING_CONFIG["min_delta"]:
                best_values[method] = val_loss
                best_states[method] = {key: value.detach().cpu().clone() for key, value in models[method].state_dict().items()}
                stale_epochs[method] = 0
            else:
                stale_epochs[method] += 1

            result_lines.append(f"{method} train={row['train_loss']:.4g}, val={val_loss:.4g}")
        print("\n".join(result_lines), flush=True)

        for method in active_methods:
            schedulers[method].step()
            if stale_epochs[method] >= TRAINING_CONFIG["patience"]:
                active[method] = False
                print(f"{method}: early stopping after {epoch} epochs; best validation loss {best_values[method]:.4g}")

    saved_histories = {}
    for method in ratio_methods:
        if best_states[method] is not None:
            models[method].load_state_dict(best_states[method])
        saved_histories[method] = save_trained_method(method, models[method], histories[method])

    print(f"[timing] grouped ratio training and save time: {format_duration(time.monotonic() - group_start)}", flush=True)
    return models, saved_histories


# ## 7. Train All Methods

# In[40]:


models = {}
histories = {}

method_index = 1
while method_index <= len(METHODS):
    method = METHODS[method_index - 1]
    if bool(TRAINING_CONFIG.get("group_ratio_methods", True)) and method in RATIO_METHODS:
        grouped_methods = []
        while method_index <= len(METHODS) and METHODS[method_index - 1] in RATIO_METHODS:
            grouped_methods.append(METHODS[method_index - 1])
            method_index += 1
        grouped_models, grouped_histories = train_ratio_methods_grouped(grouped_methods, method_index - len(grouped_methods), len(METHODS))
        models.update(grouped_models)
        histories.update(grouped_histories)
        continue

    model, history = train_method(method, method_index, len(METHODS))
    models[method] = model
    histories[method] = history
    method_index += 1

print("Trained methods:", list(models))
log_timing("trained all configured methods")


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
    parameter_grad_state = set_parameter_grad_state(model, False)
    # Iterate over test loader batches
    try:
        for batch in loader:
            batch = batch_to_device(batch)
            # Predict physical values
            log_r_pred, score_pred = ratio_score_from_gradient(model, batch["features"], batch["theta"])
            theta_physical = (
                batch["theta"].detach().cpu().numpy() * theta_scaler.scale
                + theta_scaler.mean
            )
            payload = {
                "method": method,
                "y": batch["y"].detach().cpu().numpy().ravel(),
                "likelihood_ratio_true": batch["likelihood_ratio"].detach().cpu().numpy().ravel(),
                "log_r_true": batch["log_r"].detach().cpu().numpy().ravel(),
                "log_r_pred": log_r_pred.detach().cpu().numpy().ravel(),
            }
            # Add ground truth and predicted scores for each operator parameter
            for i, name in enumerate(EFT_OPERATORS):
                payload[f"theta0_{name}"] = theta_physical[:, i]
                payload[f"score_true_{name}"] = batch["score"][:, i].detach().cpu().numpy()
                payload[f"score_pred_{name}"] = score_pred[:, i].detach().cpu().numpy()
            rows.append(pd.DataFrame(payload))
    finally:
        restore_parameter_grad_state(model, parameter_grad_state)
    # Combine predictions to a single dataframe
    return pd.concat(rows, ignore_index=True)


def collect_sallino_predictions(model: nn.Module, loader: DataLoader) -> pd.DataFrame:
    """Collect score predictions on local-score test samples."""
    rows = []
    model.eval()
    # Iterate over local test loader batches
    with torch.no_grad():
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


def binary_classification_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    """Return proper marginal hard-label risks, averaged uniformly over theta points."""
    y = predictions["y"].to_numpy(dtype=np.float64)
    log_r = predictions["log_r_pred"].to_numpy(dtype=np.float64)
    loss = np.logaddexp(0.0, log_r) - y * log_r
    probability = 1.0 / (1.0 + np.exp(-np.clip(log_r, -700.0, 700.0)))
    theta_columns = [f"theta0_{name}" for name in EFT_OPERATORS]
    metric_frame = predictions[theta_columns].copy()
    metric_frame["marginal_bce"] = loss
    per_theta = metric_frame.groupby(theta_columns, sort=False)["marginal_bce"].mean()
    return {
        "marginal_bce": float(per_theta.mean()),
        "brier": float(np.mean(np.square(probability - y))),
        "evaluation_rows": int(len(predictions)),
        "theta_points": int(len(per_theta)),
    }


def log_r_regression_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    """Return common validation diagnostics for parameterized log-r estimators."""
    values = predictions[["y", "log_r_true", "log_r_pred"]].replace([np.inf, -np.inf], np.nan).dropna()
    residual = values["log_r_pred"].to_numpy(dtype=np.float64) - values["log_r_true"].to_numpy(dtype=np.float64)
    absolute_truth = np.abs(values["log_r_true"].to_numpy(dtype=np.float64))
    trim_quantile = float(TRAINING_CONFIG.get("optuna_trim_quantile", 0.99))
    if not 0.0 < trim_quantile <= 1.0:
        raise ValueError(f"optuna_trim_quantile must lie in (0, 1], got {trim_quantile}")
    trim_threshold = float(np.quantile(absolute_truth, trim_quantile))
    trimmed = absolute_truth <= trim_threshold
    y = values["y"].to_numpy(dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "trimmed_rmse": float(np.sqrt(np.mean(np.square(residual[trimmed])))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "corr": pearson_corr(values["log_r_true"], values["log_r_pred"]),
        "y0_rmse": float(np.sqrt(np.mean(np.square(residual[y == 0.0])))),
        "y1_rmse": float(np.sqrt(np.mean(np.square(residual[y == 1.0])))),
        "trim_quantile": trim_quantile,
        "trim_threshold": trim_threshold,
        "rows": int(len(values)),
    }


if bool(TRAINING_CONFIG.get("optuna_scan_mode", False)):
    validation_rows = []
    for method in CONFIGURED_RATIO_METHODS:
        method_predictions = collect_ratio_predictions(method, models[method], ratio_loaders["validation"])
        row = {
            "method": method,
            "alpha": method_alpha(method),
        }
        row.update(binary_classification_metrics(method_predictions))
        row.update(log_r_regression_metrics(method_predictions))
        validation_rows.append(row)
    validation_metrics = pd.DataFrame(validation_rows)
    validation_path = MODEL_DIR / "validation_marginal_metrics.csv"
    validation_metrics.to_csv(validation_path, index=False)
    display(validation_metrics)
    print(f"Optuna scan mode: wrote marginal metrics to {validation_path}", flush=True)
    raise SystemExit(0)


def regenerate_predictions() -> pd.DataFrame:
    """Run inference with the currently loaded models and save test predictions."""
    if "models" not in globals():
        raise NameError("models is not defined; run the training cell before regenerating predictions.")
    prediction_frames = []
    for method in CONFIGURED_RATIO_METHODS:
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
            "target": "joint_log_r_diagnostic",
            "scope": "all rows; joint target is not marginal truth",
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "corr": pearson_corr(frame.loc[mask, "log_r_true"], frame.loc[mask, "log_r_pred"]),
        })
    for name in EFT_OPERATORS:
        true_col = f"score_true_{name}"
        pred_col = f"score_pred_{name}"
        numerator_mask = np.ones(len(frame), dtype=bool) if method == "SALLINO" else frame["y"].eq(1.0)
        mask = numerator_mask & frame[true_col].notna() & frame[pred_col].notna()
        residual = frame.loc[mask, pred_col] - frame.loc[mask, true_col]
        metric_rows.append({
            "method": method,
            "target": f"score_{name}",
            "scope": "local reference rows" if method == "SALLINO" else "numerator rows only",
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "corr": pearson_corr(frame.loc[mask, true_col], frame.loc[mask, pred_col]),
        })

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(MODEL_DIR / "test_metrics.csv", index=False)
display(metrics_df.pivot(index="method", columns="target", values=["rmse", "mae", "corr"]))

# Proper held-out risks corresponding to the implemented training objectives.
objective_rows = []
score_coordinate_scale_np = SCORE_COORDINATE_SCALE.ravel().astype(np.float64)
max_abs_log_ratio = float(TRAINING_CONFIG.get("rascal_max_abs_log_ratio", 30.0))
for method, frame in predictions.groupby("method"):
    if method == "SALLINO":
        continue
    y = frame["y"].to_numpy(dtype=np.float64)
    log_r_pred = frame["log_r_pred"].to_numpy(dtype=np.float64)
    ratio_true = np.maximum(frame["likelihood_ratio_true"].to_numpy(dtype=np.float64), np.finfo(np.float64).tiny)
    clipped_log_r = np.clip(log_r_pred, -max_abs_log_ratio, max_abs_log_ratio)
    ratio_pred = np.exp(clipped_log_r)
    classification_metrics = binary_classification_metrics(frame)
    ratio_inverse_risk = np.mean(
        (1.0 - y) * np.square(ratio_pred - ratio_true)
        + y * np.square(np.exp(-clipped_log_r) - np.reciprocal(ratio_true))
    )
    score_error = np.column_stack([
        frame[f"score_pred_{name}"].to_numpy(dtype=np.float64)
        - frame[f"score_true_{name}"].to_numpy(dtype=np.float64)
        for name in EFT_OPERATORS
    ])
    numerator_score_risk = np.mean(y * np.sum(np.square(score_error * score_coordinate_scale_np), axis=1))
    objective_rows.append({
        "method": method,
        "classifier_bce": classification_metrics["marginal_bce"],
        "brier": classification_metrics["brier"],
        "ratio_inverse_ratio_risk": float(ratio_inverse_risk),
        "numerator_score_risk": float(numerator_score_risk),
        "alpha": method_alpha(method),
    })

objective_metrics_df = pd.DataFrame(objective_rows)
objective_metrics_df.to_csv(MODEL_DIR / "test_objective_metrics.csv", index=False)
display(objective_metrics_df)


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

    return regenerate_predictions()


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
    for target, true_col, pred_col in [("joint_log_r_diagnostic", "log_r_true", "log_r_pred")]:
        plt.figure()
        for method in CONFIGURED_RATIO_METHODS:
            frame = prediction_table[prediction_table["method"] == method]
            residual = (frame[pred_col] - frame[true_col]).replace([np.inf, -np.inf], np.nan).dropna()
            plt.hist(residual, bins=80, histtype="step", density=True, label=method)
        plt.xlabel("Marginal prediction - joint log-r target")
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
            if method != "SALLINO" and "y" in frame:
                frame = frame[frame["y"] == 1.0]
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

for method in CONFIGURED_RATIO_METHODS:
    frame = predictions[predictions["method"] == method]
    plot_prediction_scatter(frame, method, "log_r_true", "log_r_pred", "joint log-r target (diagnostic)")
    score_frame = frame[frame["y"] == 1.0]
    for name in EFT_OPERATORS:
        plot_prediction_scatter(score_frame, method, f"score_true_{name}", f"score_pred_{name}", f"score_{name}")

if "SALLINO" in set(predictions["method"]):
    frame = predictions[predictions["method"] == "SALLINO"]
    for name in EFT_OPERATORS:
        plot_prediction_scatter(frame, "SALLINO", f"score_true_{name}", f"score_pred_{name}", f"score_{name}")

plot_residual_histograms(predictions)


# In[ ]:
