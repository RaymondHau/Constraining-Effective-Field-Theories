#!/usr/bin/env python
# coding: utf-8

# # EFT sample preparation
# 
# This notebook starts from the LHE files produced by `EFT_event_generation.ipynb`.
# 
# It applies configurable particle-level smearing, applies boolean event cuts, saves event and particle tables, makes diagnostic plots, splits accepted events, and writes MadMiner-style ratio and local-score sample CSVs. It intentionally stops before neural-network training.

# ## 1. Imports

# In[1]:


from __future__ import annotations

import gzip
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import Dict
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(value):
        print(value)

plt.rcParams.update({"figure.figsize": (7, 4), "axes.grid": True})


def load_workflow_config() -> Dict:
    config_path = os.environ.get("EFT_WORKFLOW_CONFIG")
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


WORKFLOW_CONFIG = load_workflow_config()


def config_section(name: str) -> Dict:
    return WORKFLOW_CONFIG.get(name, {})


# ## 2. Configuration window

# In[2]:


# -------------------------
# File layout
# -------------------------
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
WORKSPACE_DIR = LOCAL_WORKSPACE_DIR
TABLE_DIR = STORAGE_WORKSPACE_DIR / "tables"
PROCESS_DIR = LOCAL_WORKSPACE_DIR / "madgraph_work" / "processes" / "PROC_EWdim6_WBF_HAA"
LHE_ARCHIVE_DIR = STORAGE_WORKSPACE_DIR / "madgraph_work/generated_lhe_archive"
OUTPUT_DIR = TABLE_DIR / "madminer_style_training"
PLOT_DIR = OUTPUT_DIR / "plots"
LHE_FILENAME = "unweighted_events.lhe.gz"

for folder in [TABLE_DIR, OUTPUT_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Physics process
# -------------------------
FEATURE_COLUMNS = [
    "e_j1", "pt_j1", "phi_j1", "eta_j1",
    "e_j2", "pt_j2", "phi_j2", "eta_j2",
    "delta_eta_jj", "abs_delta_eta_jj", "delta_phi_jj", "abs_delta_phi_jj", "m_jj",
    "e_a1", "pt_a1", "phi_a1", "eta_a1",
    "e_a2", "pt_a2", "phi_a2", "eta_a2",
    "delta_r_aa", "pt_aa", "eta_aa", "m_aa",
    "met", "visible_ht", "zeppenfeld_aa",
]
EFT_OPERATORS = ["CWL2", "CPWL2"]
BENCHMARK_POINTS = {
    "sm":     {"CWL2": 0.0,   "CPWL2": 0.0},
    "w":      {"CWL2": 15.2,  "CPWL2": 0.1},
    "neg_w":  {"CWL2": -15.4, "CPWL2": 0.2},
    "ww":     {"CWL2": 0.3,   "CPWL2": 15.1},
    "neg_ww": {"CWL2": 0.4,   "CPWL2": -15.3},
    "w_ww":   {"CWL2": 16.88, "CPWL2": 14.95},
}
BENCHMARK_NAMES = list(BENCHMARK_POINTS)
WEIGHT_COLUMNS = [f"w_{name}" for name in BENCHMARK_NAMES]

REFERENCE_BENCHMARK = "sm"
REFERENCE_THETA = BENCHMARK_POINTS[REFERENCE_BENCHMARK]
THETA_RANGES = {"CWL2": (-16.0, 17.0), "CPWL2": (-16.0, 16.0)}
MORPHING_THETA_SCALE = {"CWL2": 16.5, "CPWL2": 16.0}
MORPHING_POLYNOMIAL_DEGREE = 2

FINAL_STATE_STATUS = 1
JET_PDGS = {1, 2, 3, 4, 5, 21}
PHOTON_PDGS = {22}
LEPTON_PDGS = {11, 13}
INVISIBLE_PDGS = {12, 14, 16}

# -------------------------
# Smearing and cuts
# -------------------------
SMEARING_SEED = 12345
SMEARING_RULES = [
    ["energy_resolution", 0.0, 0.1],
    ["pt_resolution", None, None],
    ["eta_resolution", 0.1, 0.0],
    ["phi_resolution", 0.1, 0.0],
]
# Available smearing features are defined as functions in the smearing-library cell below.

DEFAULT_CUT_CONFIG = {
    "min_jets": 2,
    "min_photons": 2,
    "min_pt_j1": 30.0,
    "min_pt_j2": 30.0,
    "max_abs_eta_j": 4.5,
    "min_pt_a1": 35.0,
    "min_pt_a2": 25.0,
    "max_abs_eta_a": 2.5,
    "m_aa_window": [122.0, 128.0],
    "min_m_jj": 250.0,
    "min_abs_delta_eta_jj": 2.0,
    "opposite_hemisphere_jets": True,
    "min_delta_r_jj": 0.4,
    "min_delta_r_aa": 0.4,
    "min_delta_r_ja": 0.4,
}
CONFIGURED_CUTS = dict(DEFAULT_CUT_CONFIG)
CUTS = []

# -------------------------
# Generated-event budgets and MadMiner-style sampling
# -------------------------
# Paper convention: sample-size numbers refer to generated events. The ratio and
# local tables below are augmented training rows derived from those generated
# event pools, and are reported separately in the summary.
EVENT_SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 2027
SAMPLE_SIZE_MODE = os.environ.get("EFT_SAMPLE_SIZE_MODE", "paper_parameterized_morphing").lower()
PAPER_PARAMETERIZED_GENERATED_EVENTS = int(os.environ.get("EFT_PAPER_PARAMETERIZED_GENERATED_EVENTS", "10000000"))
PAPER_LOCAL_SCORE_GENERATED_EVENTS = int(os.environ.get("EFT_PAPER_LOCAL_SCORE_GENERATED_EVENTS", "10000000"))
PAPER_EVALUATION_GENERATED_EVENTS = int(os.environ.get("EFT_PAPER_EVALUATION_GENERATED_EVENTS", "50000"))
QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK = int(os.environ.get("EFT_QUICK_TEST_EVENTS_PER_BENCHMARK", "100000"))
QUICK_TEST_TRAIN_ROWS = int(os.environ.get("EFT_QUICK_TEST_TRAIN_ROWS", "60000"))
QUICK_TEST_EVAL_ROWS = int(os.environ.get("EFT_QUICK_TEST_EVAL_ROWS", "12000"))

NON_REFERENCE_BENCHMARK_NAMES = [name for name in BENCHMARK_NAMES if name != REFERENCE_BENCHMARK]
if SAMPLE_SIZE_MODE == "paper_parameterized_morphing":
    events_per_basis_hypothesis = PAPER_PARAMETERIZED_GENERATED_EVENTS // (2 * len(NON_REFERENCE_BENCHMARK_NAMES))
    GENERATED_EVENTS_BY_BENCHMARK = {
        REFERENCE_BENCHMARK: events_per_basis_hypothesis * len(NON_REFERENCE_BENCHMARK_NAMES),
        **{name: events_per_basis_hypothesis for name in NON_REFERENCE_BENCHMARK_NAMES},
    }
    RATIO_SAMPLE_SIZES = {
        "train": PAPER_PARAMETERIZED_GENERATED_EVENTS,
        "validation": PAPER_EVALUATION_GENERATED_EVENTS,
        "test": PAPER_EVALUATION_GENERATED_EVENTS,
    }
    LOCAL_SCORE_SAMPLE_SIZES = {
        "train": PAPER_LOCAL_SCORE_GENERATED_EVENTS,
        "validation": PAPER_EVALUATION_GENERATED_EVENTS,
        "test": PAPER_EVALUATION_GENERATED_EVENTS,
    }
elif SAMPLE_SIZE_MODE == "quick_test":
    GENERATED_EVENTS_BY_BENCHMARK = {name: QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK for name in BENCHMARK_NAMES}
    RATIO_SAMPLE_SIZES = {"train": QUICK_TEST_TRAIN_ROWS, "validation": QUICK_TEST_EVAL_ROWS, "test": QUICK_TEST_EVAL_ROWS}
    LOCAL_SCORE_SAMPLE_SIZES = {"train": QUICK_TEST_TRAIN_ROWS, "validation": QUICK_TEST_EVAL_ROWS, "test": QUICK_TEST_EVAL_ROWS}
else:
    raise ValueError(f"Unknown SAMPLE_SIZE_MODE={SAMPLE_SIZE_MODE!r}")

# Local-score regression in the paper is SM-local: x ~ p(x|SM), t(x,z|SM).
LOCAL_SCORE_THETA_MODE = "reference"
THETA_BATCHES = {"train": 1000, "validation": 50, "test": 50} if SAMPLE_SIZE_MODE.startswith("paper") else {"train": 200, "validation": 50, "test": 50}
MAX_SAMPLE_ATTEMPTS = 25
WRITE_PARTICLE_TABLE = False
EVENT_WRITE_CHUNK_SIZE = 50_000
SAMPLE_WRITE_CHUNK_SIZE = 200_000
PROGRESS_EVERY_EVENTS = 100_000

TARGET_EPSILON = 1.0e-30
NEGATIVE_WEIGHT_POLICY = "zero"
N_EFF_FORCED = None
LOG_R_ABS_MAX = None
SCORE_COMPONENT_ABS_MAX = None
SCORE_NORM_MAX = None
DIAGNOSTIC_THETA = {"CWL2": 10.0, "CPWL2": 0.0}

_PATH_CONFIG = config_section("paths")
if _PATH_CONFIG:
    if "local_workspace" in _PATH_CONFIG:
        LOCAL_WORKSPACE_DIR = Path(_PATH_CONFIG["local_workspace"]).expanduser()
    if "storage_workspace" in _PATH_CONFIG:
        STORAGE_WORKSPACE_DIR = Path(_PATH_CONFIG["storage_workspace"]).expanduser()
        STORAGE_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR = LOCAL_WORKSPACE_DIR
    TABLE_DIR = STORAGE_WORKSPACE_DIR / _PATH_CONFIG.get("table_subdir", "tables")
    PROCESS_DIR = LOCAL_WORKSPACE_DIR / _PATH_CONFIG.get("process_subdir", "madgraph_work/processes/PROC_EWdim6_WBF_HAA")
    LHE_ARCHIVE_DIR = STORAGE_WORKSPACE_DIR / _PATH_CONFIG.get("lhe_archive_subdir", "madgraph_work/generated_lhe_archive")
    OUTPUT_DIR = TABLE_DIR / _PATH_CONFIG.get("sample_output_subdir", "madminer_style_training")
    PLOT_DIR = OUTPUT_DIR / _PATH_CONFIG.get("prepare_plot_subdir", "plots")
    LHE_FILENAME = _PATH_CONFIG.get("lhe_filename", LHE_FILENAME)

_PHYSICS_CONFIG = config_section("physics")
if _PHYSICS_CONFIG:
    FEATURE_COLUMNS = list(_PHYSICS_CONFIG.get("feature_columns", FEATURE_COLUMNS))
    EFT_OPERATORS = list(_PHYSICS_CONFIG.get("eft_operators", EFT_OPERATORS))
    BENCHMARK_POINTS = {
        name: {op: float(value) for op, value in theta.items()}
        for name, theta in _PHYSICS_CONFIG.get("benchmark_points", BENCHMARK_POINTS).items()
    }
    BENCHMARK_NAMES = list(BENCHMARK_POINTS)
    WEIGHT_COLUMNS = [f"w_{name}" for name in BENCHMARK_NAMES]
    REFERENCE_BENCHMARK = _PHYSICS_CONFIG.get("reference_benchmark", REFERENCE_BENCHMARK)
    REFERENCE_THETA = BENCHMARK_POINTS[REFERENCE_BENCHMARK]
    THETA_RANGES = {
        name: tuple(bounds)
        for name, bounds in _PHYSICS_CONFIG.get("theta_ranges", THETA_RANGES).items()
    }
    MORPHING_THETA_SCALE = dict(_PHYSICS_CONFIG.get("morphing_theta_scale", MORPHING_THETA_SCALE))
    MORPHING_POLYNOMIAL_DEGREE = int(_PHYSICS_CONFIG.get("morphing_polynomial_degree", MORPHING_POLYNOMIAL_DEGREE))

_PREP_CONFIG = config_section("preparation")
if _PREP_CONFIG:
    smearing = _PREP_CONFIG.get("smearing", {})
    SMEARING_SEED = int(smearing.get("seed", SMEARING_SEED))
    SMEARING_RULES = smearing.get("rules", SMEARING_RULES)
    CONFIGURED_CUTS = {**DEFAULT_CUT_CONFIG, **_PREP_CONFIG.get("cuts", {})}
    EVENT_SPLIT_FRACTIONS = dict(_PREP_CONFIG.get("event_split_fractions", EVENT_SPLIT_FRACTIONS))
    RANDOM_SEED = int(_PREP_CONFIG.get("random_seed", RANDOM_SEED))
    SAMPLE_SIZE_MODE = str(_PREP_CONFIG.get("sample_size_mode", SAMPLE_SIZE_MODE)).lower()
    PAPER_PARAMETERIZED_GENERATED_EVENTS = int(_PREP_CONFIG.get("paper_parameterized_generated_events", PAPER_PARAMETERIZED_GENERATED_EVENTS))
    PAPER_LOCAL_SCORE_GENERATED_EVENTS = int(_PREP_CONFIG.get("paper_local_score_generated_events", PAPER_LOCAL_SCORE_GENERATED_EVENTS))
    PAPER_EVALUATION_GENERATED_EVENTS = int(_PREP_CONFIG.get("paper_evaluation_generated_events", PAPER_EVALUATION_GENERATED_EVENTS))
    QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK = int(_PREP_CONFIG.get("quick_test_generated_events_per_benchmark", QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK))
    QUICK_TEST_TRAIN_ROWS = int(_PREP_CONFIG.get("quick_test_train_rows", QUICK_TEST_TRAIN_ROWS))
    QUICK_TEST_EVAL_ROWS = int(_PREP_CONFIG.get("quick_test_eval_rows", QUICK_TEST_EVAL_ROWS))
    LOCAL_SCORE_THETA_MODE = _PREP_CONFIG.get("local_score_theta_mode", LOCAL_SCORE_THETA_MODE)
    THETA_BATCHES = dict(_PREP_CONFIG.get("theta_batches", THETA_BATCHES))
    MAX_SAMPLE_ATTEMPTS = int(_PREP_CONFIG.get("max_sample_attempts", MAX_SAMPLE_ATTEMPTS))
    WRITE_PARTICLE_TABLE = bool(_PREP_CONFIG.get("write_particle_table", WRITE_PARTICLE_TABLE))
    EVENT_WRITE_CHUNK_SIZE = int(_PREP_CONFIG.get("event_write_chunk_size", EVENT_WRITE_CHUNK_SIZE))
    SAMPLE_WRITE_CHUNK_SIZE = int(_PREP_CONFIG.get("sample_write_chunk_size", SAMPLE_WRITE_CHUNK_SIZE))
    PROGRESS_EVERY_EVENTS = int(_PREP_CONFIG.get("progress_every_events", PROGRESS_EVERY_EVENTS))
    TARGET_EPSILON = float(_PREP_CONFIG.get("target_epsilon", TARGET_EPSILON))
    NEGATIVE_WEIGHT_POLICY = _PREP_CONFIG.get("negative_weight_policy", NEGATIVE_WEIGHT_POLICY)
    N_EFF_FORCED = _PREP_CONFIG.get("n_eff_forced", N_EFF_FORCED)
    log_r_abs_max = _PREP_CONFIG.get("log_r_abs_max", LOG_R_ABS_MAX)
    score_component_abs_max = _PREP_CONFIG.get("score_component_abs_max", SCORE_COMPONENT_ABS_MAX)
    LOG_R_ABS_MAX = None if log_r_abs_max is None else float(log_r_abs_max)
    SCORE_COMPONENT_ABS_MAX = None if score_component_abs_max is None else float(score_component_abs_max)
    SCORE_NORM_MAX = _PREP_CONFIG.get("score_norm_max", SCORE_NORM_MAX)
    DIAGNOSTIC_THETA = dict(_PREP_CONFIG.get("diagnostic_theta", DIAGNOSTIC_THETA))

NON_REFERENCE_BENCHMARK_NAMES = [name for name in BENCHMARK_NAMES if name != REFERENCE_BENCHMARK]
if SAMPLE_SIZE_MODE == "paper_parameterized_morphing":
    events_per_basis_hypothesis = PAPER_PARAMETERIZED_GENERATED_EVENTS // (2 * len(NON_REFERENCE_BENCHMARK_NAMES))
    GENERATED_EVENTS_BY_BENCHMARK = {
        REFERENCE_BENCHMARK: events_per_basis_hypothesis * len(NON_REFERENCE_BENCHMARK_NAMES),
        **{name: events_per_basis_hypothesis for name in NON_REFERENCE_BENCHMARK_NAMES},
    }
    RATIO_SAMPLE_SIZES = {
        "train": PAPER_PARAMETERIZED_GENERATED_EVENTS,
        "validation": PAPER_EVALUATION_GENERATED_EVENTS,
        "test": PAPER_EVALUATION_GENERATED_EVENTS,
    }
    LOCAL_SCORE_SAMPLE_SIZES = {
        "train": PAPER_LOCAL_SCORE_GENERATED_EVENTS,
        "validation": PAPER_EVALUATION_GENERATED_EVENTS,
        "test": PAPER_EVALUATION_GENERATED_EVENTS,
    }
elif SAMPLE_SIZE_MODE == "quick_test":
    GENERATED_EVENTS_BY_BENCHMARK = {name: QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK for name in BENCHMARK_NAMES}
    RATIO_SAMPLE_SIZES = {"train": QUICK_TEST_TRAIN_ROWS, "validation": QUICK_TEST_EVAL_ROWS, "test": QUICK_TEST_EVAL_ROWS}
    LOCAL_SCORE_SAMPLE_SIZES = {"train": QUICK_TEST_TRAIN_ROWS, "validation": QUICK_TEST_EVAL_ROWS, "test": QUICK_TEST_EVAL_ROWS}
else:
    raise ValueError(f"Unknown SAMPLE_SIZE_MODE={SAMPLE_SIZE_MODE!r}")

if _PREP_CONFIG:
    RATIO_SAMPLE_SIZES.update(_PREP_CONFIG.get("ratio_sample_sizes", {}))
    LOCAL_SCORE_SAMPLE_SIZES.update(_PREP_CONFIG.get("local_score_sample_sizes", {}))

for folder in [TABLE_DIR, OUTPUT_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ## 3. Locate generated benchmark runs

# In[ ]:


@dataclass(frozen=True)
class Benchmark:
    """One direct MadGraph benchmark and its EFT parameter point."""
    name: str
    theta: Dict[str, float]


BENCHMARKS = [Benchmark(name, theta) for name, theta in BENCHMARK_POINTS.items()]
BENCHMARK_INDEX = {benchmark.name: index for index, benchmark in enumerate(BENCHMARKS)}


def run_dirs_for_benchmark(benchmark: Benchmark) -> List[Path]:
    """Find archived, old single-run, or new multi-part generated-event directories."""
    index = BENCHMARK_INDEX[benchmark.name]
    for base in [LHE_ARCHIVE_DIR, PROCESS_DIR / "Events"]:
        parts = sorted(base.glob(f"basis_{index:02d}_{benchmark.name}_part*"))
        if parts:
            return parts
        single = base / f"basis_{index:02d}_{benchmark.name}"
        if single.exists():
            return [single]
    return [LHE_ARCHIVE_DIR / f"basis_{index:02d}_{benchmark.name}"]


RUN_DIRS = {benchmark.name: run_dirs_for_benchmark(benchmark) for benchmark in BENCHMARKS}
missing_lhe = []
for name, run_dirs in RUN_DIRS.items():
    for run_dir in run_dirs:
        lhe_path = run_dir / LHE_FILENAME
        if not lhe_path.exists():
            missing_lhe.append(lhe_path)
if missing_lhe:
    missing_text = "\n".join(str(path) for path in missing_lhe)
    raise FileNotFoundError(f"Missing generated LHE files. Run EFT_event_generation.ipynb first.\n{missing_text}")

print("Located generated LHE parts:", {name: len(paths) for name, paths in RUN_DIRS.items()})


# ## 4. LHE parsing

# In[ ]:


PARTICLE_COLUMNS = [
    "pdg_id", "status", "mother1", "mother2", "color1", "color2",
    "px", "py", "pz", "e", "mass", "lifetime", "spin",
]


def open_lhe(path: Path):
    """Open gzipped or plain LHE text."""
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else path.open("rt", encoding="utf-8", errors="replace")


def parse_lhe_events(path: Path):
    """Yield event dictionaries with particles and native reweight weights."""
    inside = False
    lines = []
    with open_lhe(path) as handle:
        for line in handle:
            # Detect the start of an event block
            if "<event" in line:
                inside = True
                lines = []
                continue
            # Detect the end of an event block and yield the parsed event
            if "</event>" in line:
                yield parse_event_block(lines)
                inside = False
                continue
            # Collect lines within the current event block
            if inside:
                lines.append(line.rstrip("\n"))


def parse_event_block(lines: List[str]) -> Dict[str, object]:
    """Parse one LHE event block."""
    # Filter out empty or whitespace-only lines
    clean = [line for line in lines if line.strip()]
    # Parse the header line for particle count and nominal weight
    header = clean[0].split()
    n_particles = int(header[0])
    nominal_weight = float(header[2])

    # Parse particle lines containing PDG IDs, status codes, and four-momenta
    particles = []
    for row in clean[1:1 + n_particles]:
        values = row.split()
        particles.append({
            "pdg_id": int(values[0]),
            "status": int(values[1]),
            "mother1": int(values[2]),
            "mother2": int(values[3]),
            "color1": int(values[4]),
            "color2": int(values[5]),
            "px": float(values[6]),
            "py": float(values[7]),
            "pz": float(values[8]),
            "e": float(values[9]),
            "mass": float(values[10]),
            "lifetime": float(values[11]),
            "spin": float(values[12]),
        })

    # Parse reweight tags containing alternative benchmark weights
    weights = {}
    text = "\n".join(clean[1 + n_particles:])
    for match in re.finditer(r"<wgt\s+id=['\"]([^'\"]+)['\"]\s*>\s*([^<]+)\s*</wgt>", text):
        weights[match.group(1)] = float(match.group(2))
    return {"nominal_weight": nominal_weight, "particles": particles, "weights": weights}


# ## 5. Smearing, observables, and cuts

# In[ ]:


def pt(px: float, py: float) -> float:
    """Return transverse momentum from Cartesian momentum components."""
    return math.hypot(px, py)


def eta(px: float, py: float, pz: float) -> float:
    """Return pseudorapidity from Cartesian momentum components."""
    p = math.sqrt(px * px + py * py + pz * pz)
    return 0.5 * math.log((p + pz + 1.0e-12) / (p - pz + 1.0e-12))


def phi(px: float, py: float) -> float:
    """Return azimuthal angle from Cartesian momentum components."""
    return math.atan2(py, px)


def delta_phi(phi_a: float, phi_b: float) -> float:
    """Return signed azimuthal-angle separation in [-pi, pi)."""
    return (phi_a - phi_b + math.pi) % (2.0 * math.pi) - math.pi


def invariant_mass(objects: Sequence[Mapping[str, float]]) -> float:
    """Return invariant mass for a collection of four-vectors."""
    # Sum the four-momentum components of all constituent objects
    e = sum(p["e"] for p in objects)
    px = sum(p["px"] for p in objects)
    py = sum(p["py"] for p in objects)
    pz = sum(p["pz"] for p in objects)
    # Calculate mass checking for unphysical negative squared values
    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


def combined_kinematics(objects: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Return aggregate four-vector kinematics for a collection of objects."""
    if not objects:
        return {"e": np.nan, "pt": np.nan, "eta": np.nan, "phi": np.nan, "mass": np.nan}
    e = sum(p["e"] for p in objects)
    px = sum(p["px"] for p in objects)
    py = sum(p["py"] for p in objects)
    pz = sum(p["pz"] for p in objects)
    return {
        "e": e,
        "pt": pt(px, py),
        "eta": eta(px, py, pz),
        "phi": phi(px, py),
        "mass": math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0)),
    }


def delta_r(obj_a: Mapping[str, float], obj_b: Mapping[str, float]) -> float:
    """Return angular separation in eta-phi space."""
    return math.hypot(obj_a["eta"] - obj_b["eta"], delta_phi(obj_a["phi"], obj_b["phi"]))


def wrap_phi(angle: float) -> float:
    """Wrap an azimuthal angle into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def momentum_from_kinematics(pt_value: float, eta_value: float, phi_value: float, energy: float, mass: float) -> Dict[str, float]:
    """Build a Cartesian four-vector from smeared pt, eta, phi, energy, and mass."""
    # Compute Cartesian momentum components from pt, eta, and phi
    px = pt_value * math.cos(phi_value)
    py = pt_value * math.sin(phi_value)
    pz = pt_value * math.sinh(eta_value)
    momentum = math.sqrt(px * px + py * py + pz * pz)
    # Ensure the energy is physically consistent with mass and momentum
    safe_energy = max(float(energy), math.sqrt(momentum * momentum + mass * mass))
    return {"px": px, "py": py, "pz": pz, "e": safe_energy}



def particle_energy(particle: Mapping[str, float]) -> float:
    """Construct particle energy from the current particle record."""
    return float(particle["e"])


def particle_px(particle: Mapping[str, float]) -> float:
    """Construct x momentum from the current particle record."""
    return float(particle["px"])


def particle_py(particle: Mapping[str, float]) -> float:
    """Construct y momentum from the current particle record."""
    return float(particle["py"])


def particle_pz(particle: Mapping[str, float]) -> float:
    """Construct z momentum from the current particle record."""
    return float(particle["pz"])


def particle_mass(particle: Mapping[str, float]) -> float:
    """Construct particle mass from the current particle record."""
    return max(float(particle["mass"]), 0.0)


def particle_pt(particle: Mapping[str, float]) -> float:
    """Construct transverse momentum from the current particle record."""
    return pt(particle["px"], particle["py"])


def particle_eta(particle: Mapping[str, float]) -> float:
    """Construct pseudorapidity from the current particle record."""
    return eta(particle["px"], particle["py"], particle["pz"])


def particle_phi(particle: Mapping[str, float]) -> float:
    """Construct azimuthal angle from the current particle record."""
    return phi(particle["px"], particle["py"])


def particle_momentum_abs(particle: Mapping[str, float]) -> float:
    """Construct the magnitude of three-momentum from the current particle record."""
    return math.sqrt(particle["px"] ** 2 + particle["py"] ** 2 + particle["pz"] ** 2)


def update_energy_consistency(particle: Dict[str, float]) -> None:
    """Keep energy at least as large as the on-shell energy implied by momentum and mass."""
    mass = particle_mass(particle)
    momentum = particle_momentum_abs(particle)
    particle["e"] = max(float(particle["e"]), math.sqrt(momentum * momentum + mass * mass))


def set_particle_energy(particle: Dict[str, float], value: float) -> None:
    """Write a smeared energy and rescale three-momentum while preserving direction."""
    mass = particle_mass(particle)
    current_momentum = particle_momentum_abs(particle)
    # Bound energy by mass and calculate the target momentum magnitude
    target_energy = max(float(value), mass)
    target_momentum = math.sqrt(max(target_energy * target_energy - mass * mass, 0.0))
    particle["e"] = target_energy
    # Rescale momentum components to match target magnitude while maintaining direction
    if current_momentum > 0.0:
        scale = target_momentum / current_momentum
        particle["px"] = float(particle["px"]) * scale
        particle["py"] = float(particle["py"]) * scale
        particle["pz"] = float(particle["pz"]) * scale


def set_particle_px(particle: Dict[str, float], value: float) -> None:
    """Write a smeared x momentum component and keep energy physically consistent."""
    particle["px"] = float(value)
    update_energy_consistency(particle)


def set_particle_py(particle: Dict[str, float], value: float) -> None:
    """Write a smeared y momentum component and keep energy physically consistent."""
    particle["py"] = float(value)
    update_energy_consistency(particle)


def set_particle_pz(particle: Dict[str, float], value: float) -> None:
    """Write a smeared z momentum component and keep energy physically consistent."""
    particle["pz"] = float(value)
    update_energy_consistency(particle)


def set_particle_mass(particle: Dict[str, float], value: float) -> None:
    """Write a smeared non-negative mass and keep energy physically consistent."""
    particle["mass"] = max(float(value), 0.0)
    update_energy_consistency(particle)


def set_particle_momentum_abs(particle: Dict[str, float], value: float) -> None:
    """Write a smeared three-momentum magnitude while preserving its current direction."""
    current = particle_momentum_abs(particle)
    target = max(float(value), 0.0)
    if current <= 0.0:
        return
    # Rescale momentum vector components
    scale = target / current
    particle["px"] = float(particle["px"]) * scale
    particle["py"] = float(particle["py"]) * scale
    particle["pz"] = float(particle["pz"]) * scale
    update_energy_consistency(particle)


def set_particle_pt(particle: Dict[str, float], value: float) -> None:
    """Write a smeared transverse momentum through the particle four-vector."""
    updated = momentum_from_kinematics(max(float(value), 0.0), particle_eta(particle), particle_phi(particle), particle_energy(particle), particle_mass(particle))
    particle.update(updated)


def set_particle_eta(particle: Dict[str, float], value: float) -> None:
    """Write a smeared pseudorapidity through the particle four-vector."""
    updated = momentum_from_kinematics(particle_pt(particle), float(value), particle_phi(particle), particle_energy(particle), particle_mass(particle))
    particle.update(updated)


def set_particle_phi(particle: Dict[str, float], value: float) -> None:
    """Write a smeared azimuthal angle through the particle four-vector."""
    updated = momentum_from_kinematics(particle_pt(particle), particle_eta(particle), wrap_phi(float(value)), particle_energy(particle), particle_mass(particle))
    particle.update(updated)


SMEARING_FEATURE_LIBRARY = {
    "energy_resolution": {"value": particle_energy, "write": set_particle_energy},
    "px_resolution": {"value": particle_px, "write": set_particle_px},
    "py_resolution": {"value": particle_py, "write": set_particle_py},
    "pz_resolution": {"value": particle_pz, "write": set_particle_pz},
    "pt_resolution": {"value": particle_pt, "write": set_particle_pt},
    "eta_resolution": {"value": particle_eta, "write": set_particle_eta},
    "phi_resolution": {"value": particle_phi, "write": set_particle_phi},
    "mass_resolution": {"value": particle_mass, "write": set_particle_mass},
    "momentum_resolution": {"value": particle_momentum_abs, "write": set_particle_momentum_abs},
}


def active_smearing_rules() -> List[Tuple[str, float, float]]:
    """Return enabled smearing rules as validated feature, absolute, relative triples."""
    rules = []
    for feature, absolute, relative in SMEARING_RULES:
        # Verify feature name matches library definition
        if feature not in SMEARING_FEATURE_LIBRARY:
            raise KeyError(f"Unknown smearing feature {feature!r}. Available features: {list(SMEARING_FEATURE_LIBRARY)}")
        # Skip disabled rules (where both resolution parameters are None)
        if absolute is None and relative is None:
            continue
        rules.append((feature, float(absolute or 0.0), float(relative or 0.0)))
    return rules


def smear_value(value: float, absolute: float, relative: float, rng: np.random.Generator) -> float:
    """Draw one Gaussian-smeared value using absolute and relative resolution terms."""
    # Compute absolute resolution width and draw from a normal distribution if positive
    sigma = absolute + relative * abs(value)
    return float(value if sigma <= 0.0 else rng.normal(value, sigma))


def apply_smearing_rule(particle: Dict[str, float], feature: str, absolute: float, relative: float, rng: np.random.Generator) -> None:
    """Construct one feature, smear it, and write it back through the feature library."""
    # Retrieve the read/write mappings, compute smeared value, and overwrite the attribute
    feature_definition = SMEARING_FEATURE_LIBRARY[feature]
    value = feature_definition["value"](particle)
    smeared_value = smear_value(value, absolute, relative, rng)
    feature_definition["write"](particle, smeared_value)


def smear_particle(particle: Mapping[str, float], rng: np.random.Generator) -> Dict[str, float]:
    """Apply configured smearing rules to one final-state particle four-vector."""
    smeared = dict(particle)
    # Only smear final state particles
    if particle["status"] != FINAL_STATE_STATUS:
        return smeared

    # Apply all active resolution rules sequentially
    for feature, absolute, relative in active_smearing_rules():
        apply_smearing_rule(smeared, feature, absolute, relative, rng)

    return smeared


def particle_kinematics(particle: Mapping[str, float]) -> Dict[str, float]:
    """Return pt, eta, and phi for one particle record."""
    return {
        "pt": pt(particle["px"], particle["py"]),
        "eta": eta(particle["px"], particle["py"], particle["pz"]),
        "phi": phi(particle["px"], particle["py"]),
    }


def event_objects(particles: List[Mapping[str, float]]) -> Dict[str, object]:
    """Return final-state object collections used by observables and cuts."""
    # Filter final-state particles and identify jets and photons by PDG code
    final = [p for p in particles if p["status"] == FINAL_STATE_STATUS]
    jets = [dict(p, **particle_kinematics(p)) for p in final if abs(p["pdg_id"]) in JET_PDGS]
    photons = [dict(p, **particle_kinematics(p)) for p in final if abs(p["pdg_id"]) in PHOTON_PDGS]
    leptons = [dict(p, **particle_kinematics(p)) for p in final if abs(p["pdg_id"]) in LEPTON_PDGS]

    # Sort jets and photons by transverse momentum in descending order to find leaders
    jets = sorted(jets, key=lambda p: p["pt"], reverse=True)
    photons = sorted(photons, key=lambda p: p["pt"], reverse=True)
    leptons = sorted(leptons, key=lambda p: p["pt"], reverse=True)
    return {
        "final": final, "jets": jets, "photons": photons, "leptons": leptons,
        "n_jets": len(jets), "n_photons": len(photons), "n_leptons": len(leptons),
    }


def reconstruct_z_systems(leptons: Sequence[Mapping[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Pair four e/mu leptons into opposite-sign same-flavour Z candidates."""
    selected = list(leptons[:4])
    if len(selected) < 4:
        return combined_kinematics([]), combined_kinematics([])
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    candidates = []
    for first, second in pairings:
        pairs = ([selected[first[0]], selected[first[1]]], [selected[second[0]], selected[second[1]]])
        if all(a["pdg_id"] == -b["pdg_id"] for a, b in pairs):
            systems = [combined_kinematics(pair) for pair in pairs]
            candidates.append((sum(abs(system["mass"] - 91.1876) for system in systems), systems))
    systems = min(candidates, key=lambda item: item[0])[1] if candidates else [
        combined_kinematics(selected[:2]), combined_kinematics(selected[2:4])
    ]
    systems.sort(key=lambda system: abs(system["mass"] - 91.1876))
    return systems[0], systems[1]


def extract_observables(particles: List[Mapping[str, float]]) -> Tuple[Dict[str, float], Dict[str, object]] | Tuple[None, Dict[str, object]]:
    """Build observables from smeared final-state particles."""
    # Group smeared particles into jets, photons, and general final state list
    objects = event_objects(particles)
    jets = objects["jets"]
    photons = objects["photons"]
    leptons = objects["leptons"]
    final = objects["final"]
    if len(jets) < 2:
        return None, objects

    # Calculate coordinate observables for jets and photons
    j1, j2 = jets[0], jets[1]
    signed_delta_phi = delta_phi(j1["phi"], j2["phi"]) * (-1.0 + 2.0 * float(j1["eta"] > j2["eta"]))
    delta_eta_jj = j1["eta"] - j2["eta"]
    dijet_mass = invariant_mass([j1, j2])

    a1 = photons[0] if len(photons) > 0 else None
    a2 = photons[1] if len(photons) > 1 else None
    diphoton = combined_kinematics(photons[:2]) if len(photons) >= 2 else combined_kinematics([])
    if len(photons) >= 2:
        diphoton_delta_r = delta_r(a1, a2)
        eta_midpoint_jj = 0.5 * (j1["eta"] + j2["eta"])
        eta_span_jj = abs(j1["eta"] - j2["eta"])
        zeppenfeld_aa = (diphoton["eta"] - eta_midpoint_jj) / eta_span_jj if eta_span_jj > 0.0 else np.nan
    else:
        diphoton_delta_r = np.nan
        zeppenfeld_aa = np.nan

    selected_leptons = list(leptons[:4])
    four_lepton = combined_kinematics(selected_leptons) if len(selected_leptons) == 4 else combined_kinematics([])
    z1, z2 = reconstruct_z_systems(selected_leptons)

    # Identify visible final state particles and calculate transverse missing energy (MET)
    visible_particles = [p for p in final if abs(p["pdg_id"]) not in INVISIBLE_PDGS]
    visible_px = sum(p["px"] for p in visible_particles)
    visible_py = sum(p["py"] for p in visible_particles)
    reconstructed_met_px = -visible_px
    reconstructed_met_py = -visible_py
    visible_ht = sum(pt(p["px"], p["py"]) for p in visible_particles)

    obs = {
        "e_j1": j1["e"],
        "pt_j1": j1["pt"],
        "phi_j1": j1["phi"],
        "eta_j1": j1["eta"],
        "e_j2": j2["e"],
        "pt_j2": j2["pt"],
        "phi_j2": j2["phi"],
        "eta_j2": j2["eta"],
        "delta_eta_jj": delta_eta_jj,
        "abs_delta_eta_jj": abs(delta_eta_jj),
        "delta_phi_jj": signed_delta_phi,
        "abs_delta_phi_jj": abs(signed_delta_phi),
        "m_jj": dijet_mass,
        "met": math.hypot(reconstructed_met_px, reconstructed_met_py),
        "visible_ht": visible_ht,
        "e_a1": a1["e"] if a1 is not None else np.nan,
        "pt_a1": a1["pt"] if a1 is not None else np.nan,
        "phi_a1": a1["phi"] if a1 is not None else np.nan,
        "eta_a1": a1["eta"] if a1 is not None else np.nan,
        "e_a2": a2["e"] if a2 is not None else np.nan,
        "pt_a2": a2["pt"] if a2 is not None else np.nan,
        "phi_a2": a2["phi"] if a2 is not None else np.nan,
        "eta_a2": a2["eta"] if a2 is not None else np.nan,
        "delta_r_aa": diphoton_delta_r,
        "pt_aa": diphoton["pt"],
        "eta_aa": diphoton["eta"],
        "m_aa": diphoton["mass"],
        "zeppenfeld_aa": zeppenfeld_aa,
    }
    for index in range(4):
        lepton = selected_leptons[index] if index < len(selected_leptons) else None
        for key in ("e", "pt", "phi", "eta"):
            obs[f"{key}_l{index + 1}"] = lepton[key] if lepton is not None else np.nan
    for key in ("e", "pt", "phi", "eta"):
        obs[f"{key}_4l"] = four_lepton[key]
        obs[f"{key}_z1"] = z1[key]
        obs[f"{key}_z2"] = z2[key]
    obs["m_4l"] = four_lepton["mass"]
    obs["m_z1"] = z1["mass"]
    obs["m_z2"] = z2["mass"]
    return obs, objects


def cut_value(config: Mapping[str, object], name: str):
    """Return a configured cut value, with None disabling optional cuts."""
    return config.get(name)


def build_configured_cuts(config: Mapping[str, object]) -> List[Callable[[Mapping[str, float], Mapping[str, object]], bool]]:
    """Build boolean analysis cuts from the JSON-compatible cut configuration."""
    cuts = []

    min_jets = cut_value(config, "min_jets")
    if min_jets is not None:
        cuts.append(lambda obs, objects, min_jets=int(min_jets): objects["n_jets"] >= min_jets)

    min_photons = cut_value(config, "min_photons")
    if min_photons is not None:
        cuts.append(lambda obs, objects, min_photons=int(min_photons): objects["n_photons"] >= min_photons)

    min_leptons = cut_value(config, "min_leptons")
    if min_leptons is not None:
        cuts.append(lambda obs, objects, min_leptons=int(min_leptons): objects["n_leptons"] >= min_leptons)

    for feature, key in [
        ("pt_j1", "min_pt_j1"),
        ("pt_j2", "min_pt_j2"),
        ("pt_a1", "min_pt_a1"),
        ("pt_a2", "min_pt_a2"),
        ("m_jj", "min_m_jj"),
        ("abs_delta_eta_jj", "min_abs_delta_eta_jj"),
        ("delta_r_aa", "min_delta_r_aa"),
    ]:
        value = cut_value(config, key)
        if value is not None:
            cuts.append(lambda obs, objects, feature=feature, value=float(value): np.isfinite(obs[feature]) and obs[feature] >= value)

    max_abs_eta_j = cut_value(config, "max_abs_eta_j")
    if max_abs_eta_j is not None:
        cuts.append(
            lambda obs, objects, max_abs_eta_j=float(max_abs_eta_j): (
                np.isfinite(obs["eta_j1"])
                and np.isfinite(obs["eta_j2"])
                and abs(obs["eta_j1"]) <= max_abs_eta_j
                and abs(obs["eta_j2"]) <= max_abs_eta_j
            )
        )

    max_abs_eta_a = cut_value(config, "max_abs_eta_a")
    if max_abs_eta_a is not None:
        cuts.append(
            lambda obs, objects, max_abs_eta_a=float(max_abs_eta_a): (
                np.isfinite(obs["eta_a1"])
                and np.isfinite(obs["eta_a2"])
                and abs(obs["eta_a1"]) <= max_abs_eta_a
                and abs(obs["eta_a2"]) <= max_abs_eta_a
            )
        )

    m_aa_window = cut_value(config, "m_aa_window")
    if m_aa_window is not None:
        low, high = float(m_aa_window[0]), float(m_aa_window[1])
        cuts.append(lambda obs, objects, low=low, high=high: np.isfinite(obs["m_aa"]) and low <= obs["m_aa"] <= high)

    if bool(cut_value(config, "opposite_hemisphere_jets")):
        cuts.append(
            lambda obs, objects: (
                np.isfinite(obs["eta_j1"])
                and np.isfinite(obs["eta_j2"])
                and obs["eta_j1"] * obs["eta_j2"] < 0.0
            )
        )

    min_delta_r_jj = cut_value(config, "min_delta_r_jj")
    if min_delta_r_jj is not None:
        cuts.append(lambda obs, objects, value=float(min_delta_r_jj): delta_r(objects["jets"][0], objects["jets"][1]) >= value)

    min_delta_r_ja = cut_value(config, "min_delta_r_ja")
    if min_delta_r_ja is not None:
        def jet_photon_separation(obs, objects, value=float(min_delta_r_ja)):
            jets = objects["jets"][:2]
            photons = objects["photons"][:2]
            return all(delta_r(jet, photon) >= value for jet in jets for photon in photons)

        cuts.append(jet_photon_separation)

    return cuts


CUTS = build_configured_cuts(CONFIGURED_CUTS)
print("Configured cuts:", CONFIGURED_CUTS)


def passes_cuts(obs: Mapping[str, float], objects: Mapping[str, object]) -> bool:
    """Return True when every configured boolean cut accepts the event."""
    return all(bool(cut(obs, objects)) for cut in CUTS)


def ordered_weights(native_weights: Mapping[str, float], nominal_weight: float) -> Dict[str, float]:
    """Map MadGraph reweight names into stable benchmark weight columns."""
    values = {}
    # Fetch matching weight value for each benchmark name (supporting standard LHA indices)
    for i, name in enumerate(BENCHMARK_NAMES, start=1):
        values[f"w_{name}"] = native_weights.get(name, native_weights.get(f"rwgt_{i}", nominal_weight))
    return values


# ## 6. Build smeared and cut event tables

# In[6]:


def write_csv_chunk(path: Path, frame: pd.DataFrame, header: bool) -> None:
    """Append a chunk to a CSV file and flush progress immediately."""
    frame.to_csv(path, mode="a", header=header, index=False)


def build_event_tables() -> pd.DataFrame:
    """Parse benchmark LHE files, smear final states, apply cuts, and save CSV tables."""
    rng = np.random.default_rng(SMEARING_SEED)
    event_chunks = []
    event_rows: List[Dict[str, object]] = []
    particle_rows: List[Dict[str, object]] = []
    event_id = 0
    event_path = TABLE_DIR / "end_to_end_events.csv"
    particle_path = TABLE_DIR / "end_to_end_particles.csv"
    if event_path.exists():
        event_path.unlink()
    if WRITE_PARTICLE_TABLE and particle_path.exists():
        particle_path.unlink()

    def flush_event_rows() -> None:
        nonlocal event_rows
        if not event_rows:
            return
        chunk = pd.DataFrame(event_rows)
        write_csv_chunk(event_path, chunk, header=not event_path.exists())
        event_chunks.append(chunk)
        print(f"Flushed {len(chunk):,} accepted event rows to {event_path}", flush=True)
        event_rows = []

    def flush_particle_rows() -> None:
        nonlocal particle_rows
        if not WRITE_PARTICLE_TABLE or not particle_rows:
            return
        chunk = pd.DataFrame(particle_rows)
        write_csv_chunk(particle_path, chunk, header=not particle_path.exists())
        print(f"Flushed {len(chunk):,} particle rows to {particle_path}", flush=True)
        particle_rows = []

    print("Active smearing rules:", active_smearing_rules())
    print("Particle table output:", "enabled" if WRITE_PARTICLE_TABLE else "disabled")
    for benchmark in BENCHMARKS:
        for part_index, run_dir in enumerate(RUN_DIRS[benchmark.name], start=1):
            lhe_path = run_dir / LHE_FILENAME
            accepted_in_part = 0
            parsed_in_part = 0
            print("Reading", benchmark.name, f"part {part_index}/{len(RUN_DIRS[benchmark.name])}", lhe_path, flush=True)
            for local_event_id, event in enumerate(parse_lhe_events(lhe_path)):
                parsed_in_part = local_event_id + 1
                smeared_particles = [smear_particle(particle, rng) for particle in event["particles"]]
                obs, objects = extract_observables(smeared_particles)
                if obs is None or not passes_cuts(obs, objects):
                    if PROGRESS_EVERY_EVENTS > 0 and (local_event_id + 1) % PROGRESS_EVERY_EVENTS == 0:
                        print(
                            f"{benchmark.name} part {part_index}: parsed {local_event_id + 1:,}, "
                            f"accepted {accepted_in_part:,}, total accepted {event_id:,}",
                            flush=True,
                        )
                    continue

                weights = ordered_weights(event["weights"], event["nominal_weight"])
                event_rows.append({
                    "event_id": event_id,
                    "source_benchmark": benchmark.name,
                    "source_part": part_index,
                    "local_event_id": local_event_id,
                    **obs,
                    **weights,
                })

                if WRITE_PARTICLE_TABLE:
                    for particle_index, particle in enumerate(smeared_particles):
                        particle_rows.append({
                            "event_id": event_id,
                            "source_benchmark": benchmark.name,
                            "source_part": part_index,
                            "particle_index": particle_index,
                            **particle,
                        })
                event_id += 1
                accepted_in_part += 1

                if len(event_rows) >= EVENT_WRITE_CHUNK_SIZE:
                    flush_event_rows()
                if WRITE_PARTICLE_TABLE and len(particle_rows) >= EVENT_WRITE_CHUNK_SIZE:
                    flush_particle_rows()
                if PROGRESS_EVERY_EVENTS > 0 and (local_event_id + 1) % PROGRESS_EVERY_EVENTS == 0:
                    print(
                        f"{benchmark.name} part {part_index}: parsed {local_event_id + 1:,}, "
                        f"accepted {accepted_in_part:,}, total accepted {event_id:,}",
                        flush=True,
                    )
            print(
                f"Finished {benchmark.name} part {part_index}: parsed {parsed_in_part:,}, "
                f"accepted {accepted_in_part:,}, total accepted {event_id:,}",
                flush=True,
            )

    flush_event_rows()
    flush_particle_rows()
    if not event_chunks:
        raise RuntimeError("No events passed the configured cuts")
    return pd.concat(event_chunks, ignore_index=True)


event_df = build_event_tables()
before_clean = len(event_df)
event_df = event_df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + WEIGHT_COLUMNS).reset_index(drop=True)
if len(event_df) != before_clean:
    print(f"Dropped {before_clean - len(event_df):,} events with non-finite features or weights", flush=True)
    event_df.to_csv(TABLE_DIR / "end_to_end_events.csv", index=False)
print(f"Accepted events: {len(event_df):,}")
display(event_df.head())


# ## 7. Direct sample plots

# In[7]:


def normalized_density(values: np.ndarray, bins: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return a unit-area histogram density."""
    # Calculate bin heights and normalize by total integrated area to form a PDF
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    area = np.sum(hist * np.diff(bins))
    return hist / area if area > 0.0 else hist


def plot_feature_distributions() -> None:
    """Plot direct benchmark distributions for configured training features."""
    # Loop over configured feature columns
    for feature in FEATURE_COLUMNS:
        values = event_df[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        low, high = np.percentile(values, [1, 99])
        if np.isclose(low, high):
            print(f"Skipping {feature}: nearly constant at {low:.6g}")
            continue

        # Setup bin spacing and histogram centers
        bins = np.linspace(low, high, 50)
        centers = 0.5 * (bins[:-1] + bins[1:])
        plt.figure()

        # Plot normalized histogram shapes for each direct benchmark run
        for name in BENCHMARK_NAMES:
            sample = event_df.loc[event_df["source_benchmark"] == name, feature].to_numpy()
            plt.step(centers, normalized_density(sample, bins), where="mid", label=name)
        plt.xlabel(feature)
        plt.ylabel("Normalized events")
        plt.legend()
        plt.tight_layout()
        plt.show()


plot_feature_distributions()


# ## 8. Morphing setup

# In[ ]:


def theta_vector(theta: Mapping[str, float]) -> np.ndarray:
    """Return theta values ordered to match EFT_OPERATORS."""
    # Convert a dictionary representation of theta to a sorted list of floats
    return np.array([theta[name] for name in EFT_OPERATORS], dtype=np.float64)


def theta_matrix(thetas: Sequence[Mapping[str, float]]) -> np.ndarray:
    """Stack theta dictionaries into a two-dimensional array."""
    # Convert a sequence of parameter points to a matrix (n_points, n_parameters)
    return np.vstack([theta_vector(theta) for theta in thetas])


def polynomial_exponents(degree: int) -> List[Tuple[int, int]]:
    """Return all two-parameter monomial exponents up to a total degree."""
    return [(power_1, total - power_1) for total in range(degree + 1) for power_1 in range(total + 1)]


def morphing_basis(theta: np.ndarray) -> np.ndarray:
    """Return the configured EFT polynomial basis in scaled coordinates."""
    scale = theta_vector(MORPHING_THETA_SCALE)
    scaled_theta = np.asarray(theta, dtype=np.float64) / scale
    c1 = scaled_theta[..., 0]
    c2 = scaled_theta[..., 1]
    return np.stack([c1**power_1 * c2**power_2 for power_1, power_2 in polynomial_exponents(MORPHING_POLYNOMIAL_DEGREE)], axis=-1)


def morphing_basis_gradient(theta: np.ndarray) -> np.ndarray:
    """Return gradients of the configured basis with respect to physical theta."""
    scale = theta_vector(MORPHING_THETA_SCALE)
    scaled_theta = np.asarray(theta, dtype=np.float64) / scale
    c1 = scaled_theta[..., 0]
    c2 = scaled_theta[..., 1]
    exponents = polynomial_exponents(MORPHING_POLYNOMIAL_DEGREE)
    grad_c1 = np.stack([np.zeros_like(c1) if i == 0 else i * c1 ** (i - 1) * c2**j for i, j in exponents], axis=-1) / scale[0]
    grad_c2 = np.stack([np.zeros_like(c2) if j == 0 else j * c1**i * c2 ** (j - 1) for i, j in exponents], axis=-1) / scale[1]
    return np.stack([grad_c1, grad_c2], axis=-2)


quadratic_basis = morphing_basis
quadratic_basis_gradient = morphing_basis_gradient


#Convert benchmark vectors to matricies
benchmark_theta = theta_matrix([BENCHMARK_POINTS[name] for name in BENCHMARK_NAMES])

#Calculate the quadratic polynomial components for each benchmark point
basis_at_benchmarks = morphing_basis(benchmark_theta)
if basis_at_benchmarks.shape[0] < basis_at_benchmarks.shape[1]:
    raise ValueError(f"Morphing degree {MORPHING_POLYNOMIAL_DEGREE} requires {basis_at_benchmarks.shape[1]} benchmark points")

# checks condition number of the basis matrix, we want a low condition number (e.g. < 10) for stability
print("Morphing basis condition number:", f"{np.linalg.cond(basis_at_benchmarks):.3g}")

# Compute the pseudo-inverse to obtain the morphing matrix.
morphing_matrix = np.linalg.pinv(basis_at_benchmarks)

# Convert dataframe columns of benchmark weights to a NumPy array
benchmark_weights = event_df[WEIGHT_COLUMNS].to_numpy(dtype=np.float64)

# Extract the polynomial coefficients pr event
event_coefficients = benchmark_weights @ morphing_matrix.T

# Estimate the total cross section at each benchmark point
# sum of weight/generated events
sigma_benchmarks = np.array([
    event_df.loc[event_df["source_benchmark"] == name, f"w_{name}"].sum() / GENERATED_EVENTS_BY_BENCHMARK[name]
    for name in BENCHMARK_NAMES
], dtype=np.float64)

# Solve for the polynomial coefficients of the total cross section
sigma_coefficients = morphing_matrix @ sigma_benchmarks

# Predict the total cross section at the reference parameter
reference_theta = theta_vector(REFERENCE_THETA)
reference_basis = morphing_basis(reference_theta[None, :])[0]
reference_sigma = float(reference_basis @ sigma_coefficients)

# Compute the proposal mixture density proxy.

benchmark_counts = event_df["source_benchmark"].value_counts().reindex(BENCHMARK_NAMES).to_numpy(dtype=np.float64)
benchmark_mixture_fractions = benchmark_counts / benchmark_counts.sum()
proposal_density_proxy = np.zeros(len(event_df), dtype=np.float64)
for fraction, name, sigma in zip(benchmark_mixture_fractions, BENCHMARK_NAMES, sigma_benchmarks):
    # Accumulate the weighted contribution of each benchmark's cross-section-normalized weight
    proposal_density_proxy += fraction * event_df[f"w_{name}"].to_numpy(dtype=np.float64) / max(float(sigma), TARGET_EPSILON)
proposal_density_proxy = np.maximum(proposal_density_proxy, TARGET_EPSILON)

#prints cross section at benchmark and refrence points
print("Reference sigma:", reference_sigma)
print("Benchmark sigma estimates:", {name: f"{sigma:.6e}" for name, sigma in zip(BENCHMARK_NAMES, sigma_benchmarks)})

#print how many accepted events are from each benchmark /total number of accepted events
print("Accepted-event mixture fractions:", dict(zip(BENCHMARK_NAMES, np.round(benchmark_mixture_fractions, 4))))


# ## 9. Target builders

# In[9]:


def morphed_weights(indices: np.ndarray, theta_values: np.ndarray) -> np.ndarray:
    """Return morphed event weights for selected events and theta values."""
    # Project event weights to target points using event basis coefficients
    phi = morphing_basis(theta_values)
    return np.einsum("ij,ij->i", event_coefficients[indices], phi)


def morphed_sigma(theta_values: np.ndarray) -> np.ndarray:
    """Return morphed total cross sections for theta values."""
    # Project total cross sections using basis representation
    return morphing_basis(theta_values) @ sigma_coefficients


def score_at_theta(indices: np.ndarray, theta_values: np.ndarray) -> np.ndarray:
    """Return joint score t(x,z|theta) for selected event-theta pairs."""
    coeffs = event_coefficients[indices]
    phi = morphing_basis(theta_values)
    grad_phi = morphing_basis_gradient(theta_values)

    # Calculate morphed weights, total cross-sections, and their parameter gradients
    weights_theta = np.einsum("ij,ij->i", coeffs, phi)
    sigma_theta = phi @ sigma_coefficients
    grad_w = np.einsum("ij,ikj->ik", coeffs, grad_phi)
    grad_sigma = np.einsum("j,ikj->ik", sigma_coefficients, grad_phi)

    # Protect against divide-by-zero errors in unphysical regions
    weights_safe = np.maximum(weights_theta, TARGET_EPSILON)
    sigma_safe = np.maximum(sigma_theta, TARGET_EPSILON)

    # Compute local score vector t(x, z | theta) = grad_theta log p(x, z | theta)
    return grad_w / weights_safe[:, None] - grad_sigma / sigma_safe[:, None]


def log_ratio_and_score(indices: np.ndarray, theta0_values: np.ndarray, theta1_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return log ratio, ratio, soft class probability, and score at theta0."""
    # Calculate morphed event weights and cross sections at numerator and denominator
    w0 = morphed_weights(indices, theta0_values)
    w1 = morphed_weights(indices, theta1_values)
    sigma0 = morphed_sigma(theta0_values)
    sigma1 = morphed_sigma(theta1_values)

    # Identify where parameters lie inside valid positive weight/cross section boundaries
    valid = (w0 > TARGET_EPSILON) & (w1 > TARGET_EPSILON) & (sigma0 > TARGET_EPSILON) & (sigma1 > TARGET_EPSILON)

    # Compute likelihood ratios and score vectors
    log_r = np.log(np.maximum(w0, TARGET_EPSILON)) - np.log(np.maximum(w1, TARGET_EPSILON)) + np.log(np.maximum(sigma1, TARGET_EPSILON)) - np.log(np.maximum(sigma0, TARGET_EPSILON))
    ratio = np.exp(np.clip(log_r, -30.0, 30.0))
    soft_y = ratio / (1.0 + ratio)
    score = score_at_theta(indices, theta0_values)

    # Replace invalid regions with NaN to prevent training corruption
    log_r = np.where(valid, log_r, np.nan)
    ratio = np.where(valid, ratio, np.nan)
    soft_y = np.where(valid, soft_y, np.nan)
    score = np.where(valid[:, None], score, np.nan)
    return log_r.astype(np.float32), ratio.astype(np.float32), soft_y.astype(np.float32), score.astype(np.float32)


def target_valid_mask(log_r: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Return a finite-target mask, with optional explicitly configured truncation."""
    return finite_target_mask(log_r, score, LOG_R_ABS_MAX, SCORE_COMPONENT_ABS_MAX, SCORE_NORM_MAX)


def finite_target_mask(
    log_r: np.ndarray,
    score: np.ndarray,
    log_r_abs_max: float | None = None,
    score_component_abs_max: float | None = None,
    score_norm_max: float | None = None,
) -> np.ndarray:
    """Select finite targets, with optional explicitly requested truncation."""
    mask = np.isfinite(log_r) & np.all(np.isfinite(score), axis=1)
    if log_r_abs_max is not None:
        mask &= np.abs(log_r) <= log_r_abs_max
    if score_component_abs_max is not None:
        mask &= np.all(np.abs(score) <= score_component_abs_max, axis=1)
    if score_norm_max is not None:
        mask &= np.linalg.norm(score, axis=1) <= score_norm_max
    return mask


def draw_uniform_reference_events(
    candidate_indices: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample direct reference events without an inappropriate mixture correction."""
    positions = rng.choice(len(candidate_indices), size=n_samples, replace=True)
    return candidate_indices[positions]


# ## 10. Morphing diagnostic plots

# In[10]:


def morphed_event_probabilities(theta: Mapping[str, float] | np.ndarray, event_indices: np.ndarray | None = None) -> np.ndarray:
    """Return proposal-corrected probabilities for drawing events from the mixed benchmark pool."""
    # Translate parameter representation
    theta_arr = theta_vector(theta) if isinstance(theta, Mapping) else np.asarray(theta, dtype=np.float64)
    indices = np.arange(len(event_df)) if event_indices is None else np.asarray(event_indices)
    theta_values = np.repeat(theta_arr[None, :], len(indices), axis=0)

    # Compute proposal-weighted probability ratios for event selection
    target_density = morphed_weights(indices, theta_values) / max(float(morphed_sigma(theta_arr[None, :])[0]), TARGET_EPSILON)
    probabilities = target_density / proposal_density_proxy[indices]
    if NEGATIVE_WEIGHT_POLICY == "zero":
        probabilities = np.where(probabilities > 0.0, probabilities, 0.0)
    total = probabilities.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(len(indices), dtype=np.float64) / len(indices)
    return probabilities / total


def plot_sm_vs_morphed(theta: Mapping[str, float]) -> None:
    """Compare direct SM feature shapes to morphed target shapes."""
    # Generate event probabilities at target theta
    target_probabilities = morphed_event_probabilities(theta)
    for feature in FEATURE_COLUMNS:
        values = event_df[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        low, high = np.percentile(values, [0, 99])
        if np.isclose(low, high):
            print(f"Skipping {feature}: nearly constant at {low:.6g}")
            continue
        bins = np.linspace(low, high, 50)
        centers = 0.5 * (bins[:-1] + bins[1:])
        sm_values = event_df.loc[event_df["source_benchmark"] == REFERENCE_BENCHMARK, feature].to_numpy()

        # Setup distribution comparison plots
        plt.figure()
        plt.step(centers, normalized_density(sm_values, bins), where="mid", label="SM direct")
        plt.step(centers, normalized_density(event_df[feature].to_numpy(), bins, weights=target_probabilities), where="mid", label=f"Morphed {theta}")
        plt.xlabel(feature)
        plt.ylabel("Normalized events")
        plt.legend()
        plt.tight_layout()
        plt.show()


plot_sm_vs_morphed(DIAGNOSTIC_THETA)


# ## 11. Split and sample events

# In[11]:


def sample_theta_values(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample theta values uniformly from the configured morphing prior."""
    columns = []
    # Draw random parameter values from uniform priors for each active operator
    for name in EFT_OPERATORS:
        low, high = THETA_RANGES[name]
        columns.append(rng.uniform(low, high, size=n))
    return np.stack(columns, axis=1).astype(np.float32)


def split_event_indices(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Split base generated events into train, validation, and test partitions."""
    if not np.isclose(sum(EVENT_SPLIT_FRACTIONS.values()), 1.0):
        raise ValueError("EVENT_SPLIT_FRACTIONS must sum to one")
    # Shuffle indices and assign disjoint subsets to train/val/test splits
    shuffled = rng.permutation(len(event_df))
    train_stop = int(round(EVENT_SPLIT_FRACTIONS["train"] * len(shuffled)))
    validation_stop = train_stop + int(round(EVENT_SPLIT_FRACTIONS["validation"] * len(shuffled)))
    return {
        "train": shuffled[:train_stop],
        "validation": shuffled[train_stop:validation_stop],
        "test": shuffled[validation_stop:],
    }


def event_probabilities(theta: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    """Return proposal-corrected probabilities within one train/validation/test split."""
    # Calculate proposal weight density ratios inside current index slice
    theta_values = np.repeat(theta[None, :], len(candidate_indices), axis=0)
    target_density = morphed_weights(candidate_indices, theta_values) / max(float(morphed_sigma(theta[None, :])[0]), TARGET_EPSILON)
    probabilities = target_density / proposal_density_proxy[candidate_indices]

    # Suppress negative and excessive density weight values
    if NEGATIVE_WEIGHT_POLICY == "zero":
        probabilities = np.where(probabilities > 0.0, probabilities, 0.0)
    if N_EFF_FORCED is not None:
        probabilities = np.where(probabilities <= 1.0 / N_EFF_FORCED, probabilities, 0.0)
    total = probabilities.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(len(candidate_indices), dtype=np.float64) / len(candidate_indices)
    return probabilities / total


def draw_events_for_theta_grid(theta_grid: np.ndarray, n_samples: int, candidate_indices: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Draw many events from each theta point after computing each probability vector once."""
    examples_per_theta = int(math.ceil(n_samples / len(theta_grid)))
    chosen_events = []
    chosen_thetas = []

    # Sample events at each grid point using corresponding proposal weight correction
    for theta in theta_grid:
        remaining = n_samples - sum(len(chunk) for chunk in chosen_events)
        if remaining <= 0:
            break
        draw_count = min(examples_per_theta, remaining)
        probabilities = event_probabilities(theta.astype(np.float64), candidate_indices)
        positions = rng.choice(len(candidate_indices), size=draw_count, replace=True, p=probabilities)
        chosen_events.append(candidate_indices[positions])
        chosen_thetas.append(np.repeat(theta[None, :], draw_count, axis=0))
    return np.concatenate(chosen_events), np.concatenate(chosen_thetas, axis=0).astype(np.float32)


# ## 12. Write MadMiner-style training tables

# In[12]:


def assemble_base_columns(event_indices: np.ndarray) -> pd.DataFrame:
    """Return observables and event metadata for selected event indices."""
    # Populate columns containing base index maps and observables
    selected = event_df.iloc[event_indices].reset_index(drop=True)
    frame = pd.DataFrame({
        "event_id": selected["event_id"].to_numpy(),
        "source_benchmark": selected["source_benchmark"].to_numpy(),
        "sampled_event_index": event_indices,
    })
    for feature in FEATURE_COLUMNS:
        frame[feature] = selected[feature].to_numpy(dtype=np.float32)
    return frame


def assemble_ratio_frame(split: str, event_indices: np.ndarray, theta0: np.ndarray, theta1: np.ndarray, y: np.ndarray, soft_y: np.ndarray, log_r: np.ndarray, ratio: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Assemble one ratio-sample DataFrame."""
    # Build ratio training dataframe by merging coordinate maps and target outputs
    frame = assemble_base_columns(event_indices)
    frame.insert(0, "split", split)
    frame["y"] = y.astype(np.float32)
    frame["soft_y"] = soft_y.astype(np.float32)
    for i, name in enumerate(EFT_OPERATORS):
        frame[f"theta0_{name}"] = theta0[:, i].astype(np.float32)
        frame[f"theta1_{name}"] = theta1[:, i].astype(np.float32)
        frame[f"score_{name}"] = score[:, i].astype(np.float32)
    frame["log_r"] = log_r.astype(np.float32)
    frame["likelihood_ratio"] = ratio.astype(np.float32)
    return frame


def assemble_local_frame(split: str, event_indices: np.ndarray, theta: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Assemble one local-score DataFrame."""
    # Build local score training dataframe by merging coordinates and score gradients
    frame = assemble_base_columns(event_indices)
    frame.insert(0, "split", split)
    for i, name in enumerate(EFT_OPERATORS):
        frame[f"theta_{name}"] = theta[:, i].astype(np.float32)
        frame[f"score_{name}"] = score[:, i].astype(np.float32)
    return frame


def balanced_ratio_indices(mask: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return shuffled valid row indices with equal numerator and denominator labels."""
    # Fetch positive and negative class indices within selection mask
    valid = np.flatnonzero(mask)
    positive = valid[y[valid] == 1.0]
    negative = valid[y[valid] == 0.0]

    # Select matching numbers of positive and negative class samples to balance training targets
    keep_per_class = min(len(positive), len(negative))
    if keep_per_class == 0:
        return np.array([], dtype=np.int64)
    selected = np.concatenate([
        rng.choice(positive, size=keep_per_class, replace=False),
        rng.choice(negative, size=keep_per_class, replace=False),
    ])
    rng.shuffle(selected)
    return selected


def sample_train_ratio_like(split: str, n_samples: int, candidate_indices: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Create a MadMiner-like ratio sample with balanced theta0 and theta1 events."""
    if n_samples % 2 != 0:
        raise ValueError("Ratio samples should be even so numerator and denominator classes are balanced")
    half = n_samples // 2
    n_theta = min(THETA_BATCHES[split], half)

    # Sample grid of numerator thetas and target benchmark theta
    theta0_grid = sample_theta_values(n_theta, rng)
    theta1_grid = np.repeat(reference_theta[None, :], 1, axis=0).astype(np.float32)

    # Generate event lists for numerator and denominator splits
    numerator_indices, theta0_numerator = draw_events_for_theta_grid(theta0_grid, half, candidate_indices, rng)
    denominator_indices, _theta1_denominator = draw_events_for_theta_grid(theta1_grid, half, candidate_indices, rng)

    # Compile combined event lists and labels
    event_indices = np.concatenate([numerator_indices, denominator_indices])
    theta0_values = np.concatenate([theta0_numerator, theta0_numerator])
    theta1_values = np.repeat(reference_theta[None, :], n_samples, axis=0).astype(np.float32)
    y = np.concatenate([np.ones(half, dtype=np.float32), np.zeros(half, dtype=np.float32)])

    # Calculate training targets and clean up outliers
    log_r, ratio, soft_y, score = log_ratio_and_score(event_indices, theta0_values.astype(np.float64), theta1_values.astype(np.float64))
    mask = target_valid_mask(log_r, score)
    selected = balanced_ratio_indices(mask, y, rng)
    return assemble_ratio_frame(split, event_indices[selected], theta0_values[selected], theta1_values[selected], y[selected], soft_y[selected], log_r[selected], ratio[selected], score[selected])


def sample_train_local_like(split: str, n_samples: int, candidate_indices: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Create a local score sample. Default follows the paper: x ~ p(x|SM), t(x,z|SM)."""
    local_benchmark = REFERENCE_BENCHMARK if LOCAL_SCORE_THETA_MODE == "reference" else LOCAL_SCORE_THETA_MODE
    if local_benchmark in BENCHMARK_POINTS:
        local_theta = theta_vector(BENCHMARK_POINTS[local_benchmark])
        theta = np.repeat(local_theta[None, :], n_samples, axis=0).astype(np.float32)
        # The caller already restricts candidates to directly generated reference
        # events, so applying the mixed-proposal correction again would bias p(x|SM).
        event_indices = draw_uniform_reference_events(candidate_indices, n_samples, rng)
    else:
        n_theta = min(THETA_BATCHES[split], n_samples)
        theta_grid = sample_theta_values(n_theta, rng)
        event_indices, theta = draw_events_for_theta_grid(theta_grid, n_samples, candidate_indices, rng)

    score = score_at_theta(event_indices, theta.astype(np.float64)).astype(np.float32)
    log_r = np.zeros(n_samples, dtype=np.float32)
    mask = target_valid_mask(log_r, score)
    return assemble_local_frame(split, event_indices[mask], theta[mask], score[mask])


def build_until_enough(builder: Callable, split: str, n_samples: int, candidate_indices: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Oversample candidates until enough valid MadMiner-style rows are available."""
    chunks = []
    kept = 0
    kept_positive = 0
    kept_negative = 0
    target_per_ratio_class = n_samples // 2 if builder is sample_train_ratio_like else None
    attempts = 0

    # Loop until enough valid events have been collected
    while kept < n_samples and attempts < MAX_SAMPLE_ATTEMPTS:
        remaining = n_samples - kept
        candidate_size = int(math.ceil(1.25 * remaining))
        if builder is sample_train_ratio_like and candidate_size % 2 != 0:
            candidate_size += 1
        candidate = builder(split, candidate_size, candidate_indices, rng)

        # Enforce equal positive/negative counts for ratio training splits
        if builder is sample_train_ratio_like:
            need_positive = target_per_ratio_class - kept_positive
            need_negative = target_per_ratio_class - kept_negative
            positive_chunk = candidate[candidate["y"] == 1.0].iloc[:need_positive]
            negative_chunk = candidate[candidate["y"] == 0.0].iloc[:need_negative]
            chunk = pd.concat([positive_chunk, negative_chunk], ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)
            kept_positive += int((chunk["y"] == 1.0).sum())
            kept_negative += int((chunk["y"] == 0.0).sum())
        else:
            chunk = candidate.iloc[:remaining].copy()
        chunks.append(chunk)
        kept += len(chunk)
        attempts += 1
        print(f"{split} {builder.__name__}: candidate {len(candidate):,}, kept {kept:,}/{n_samples:,} after attempt {attempts}")

    # Concatenate chunks and confirm target counts are satisfied
    result = pd.concat(chunks, ignore_index=True)
    if len(result) < n_samples:
        raise RuntimeError(f"Only built {len(result)} valid rows for {split}")
    if builder is sample_train_ratio_like:
        counts = result["y"].value_counts().to_dict()
        if counts.get(1.0, 0) != target_per_ratio_class or counts.get(0.0, 0) != target_per_ratio_class:
            raise RuntimeError(f"Built imbalanced ratio rows for {split}: {counts}")
    return result



def even_count(n: int) -> int:
    """Return an even positive row count for balanced ratio sampling."""
    return max(2, int(n) - (int(n) % 2))


def update_sample_stats(stats: Dict[str, object], frame: pd.DataFrame, is_ratio: bool) -> None:
    """Update lightweight summary counters for a written sample chunk."""
    stats["rows"] = int(stats["rows"]) + len(frame)
    stats["unique_events"].update(frame["event_id"].astype(np.int64).tolist())
    if is_ratio:
        stats["ratio_y0_rows"] = int(stats["ratio_y0_rows"]) + int((frame["y"] == 0.0).sum())
        stats["ratio_y1_rows"] = int(stats["ratio_y1_rows"]) + int((frame["y"] == 1.0).sum())


def write_sample_in_chunks(
    builder: Callable,
    split: str,
    n_samples: int,
    candidate_indices: np.ndarray,
    rng: np.random.Generator,
    output_path: Path,
) -> Dict[str, object]:
    """Build a large training sample in bounded-size chunks and append it to CSV."""
    if output_path.exists():
        output_path.unlink()
    is_ratio = builder is sample_train_ratio_like
    written = 0
    chunk_index = 0
    stats: Dict[str, object] = {
        "rows": 0,
        "ratio_y0_rows": 0,
        "ratio_y1_rows": 0,
        "unique_events": set(),
    }

    while written < n_samples:
        remaining = n_samples - written
        chunk_target = min(SAMPLE_WRITE_CHUNK_SIZE, remaining)
        if is_ratio:
            chunk_target = even_count(chunk_target)
            if chunk_target > remaining:
                chunk_target = remaining - (remaining % 2)
            if chunk_target <= 0:
                break

        chunk = build_until_enough(builder, split, chunk_target, candidate_indices, rng)
        write_csv_chunk(output_path, chunk, header=written == 0)
        update_sample_stats(stats, chunk, is_ratio)
        written += len(chunk)
        chunk_index += 1
        print(
            f"Wrote {output_path.name} chunk {chunk_index}: {len(chunk):,} rows "
            f"({written:,}/{n_samples:,})",
            flush=True,
        )
        del chunk

    if written != n_samples:
        raise RuntimeError(f"Wrote {written:,} rows to {output_path}, expected {n_samples:,}")
    stats["unique_events"] = len(stats["unique_events"])
    return stats


rng = np.random.default_rng(RANDOM_SEED)
event_pools = split_event_indices(rng)
sample_summary_rows = []

print("Sample-size mode:", SAMPLE_SIZE_MODE)
print("Generated-event budgets:", GENERATED_EVENTS_BY_BENCHMARK)
print("Augmented ratio row requests:", RATIO_SAMPLE_SIZES)
print("Augmented local row requests:", LOCAL_SCORE_SAMPLE_SIZES)
print("Sample write chunk size:", SAMPLE_WRITE_CHUNK_SIZE)

for split in ["train", "validation", "test"]:
    split_indices = event_pools[split]
    ratio_n = even_count(RATIO_SAMPLE_SIZES[split])
    local_n = int(LOCAL_SCORE_SAMPLE_SIZES[split])
    ratio_path = OUTPUT_DIR / f"ratio_{split}.csv"
    local_path = OUTPUT_DIR / f"local_{split}.csv"
    print(f"Building {split} ratio sample: {ratio_n:,} rows from {len(split_indices):,} base events", flush=True)
    ratio_stats = write_sample_in_chunks(sample_train_ratio_like, split, ratio_n, split_indices, rng, ratio_path)

    local_benchmark = REFERENCE_BENCHMARK if LOCAL_SCORE_THETA_MODE == "reference" else LOCAL_SCORE_THETA_MODE
    if local_benchmark in BENCHMARK_POINTS:
        source = event_df.iloc[split_indices]["source_benchmark"].to_numpy()
        local_indices = split_indices[source == local_benchmark]
        if len(local_indices) == 0:
            raise RuntimeError(f"No reference-benchmark events available for local score split {split}")
    else:
        local_indices = split_indices
    print(f"Building {split} local-score sample: {local_n:,} rows from {len(local_indices):,} base events", flush=True)
    local_stats = write_sample_in_chunks(sample_train_local_like, split, local_n, local_indices, rng, local_path)

    sample_summary_rows.append({
        "split": split,
        "base_accepted_events": len(split_indices),
        "ratio_rows": int(ratio_stats["rows"]),
        "local_rows": int(local_stats["rows"]),
        "requested_ratio_rows": ratio_n,
        "requested_local_rows": local_n,
        "ratio_y0_rows": int(ratio_stats["ratio_y0_rows"]),
        "ratio_y1_rows": int(ratio_stats["ratio_y1_rows"]),
        "unique_ratio_events": int(ratio_stats["unique_events"]),
        "unique_local_events": int(local_stats["unique_events"]),
    })
    print(f"Finished {split}: ratio={ratio_stats['rows']:,}, local={local_stats['rows']:,}", flush=True)

sample_metadata = pd.DataFrame([
    {"benchmark": name, "generated_events_requested": GENERATED_EVENTS_BY_BENCHMARK[name]}
    for name in BENCHMARK_NAMES
])
sample_metadata.to_csv(OUTPUT_DIR / "generated_event_budget.csv", index=False)
pd.DataFrame(sample_summary_rows).to_csv(OUTPUT_DIR / "sample_summary.csv", index=False)


# ## 13. Output summary

# In[13]:


if "OUTPUT_DIR" not in globals():
    NOTEBOOK_DIR = Path.cwd()
    OUTPUT_DIR = NOTEBOOK_DIR / "barebones_eft_workspace" / "tables" / "madminer_style_training"

if "sample_summary_rows" in globals():
    summary_df = pd.DataFrame(sample_summary_rows)
elif (OUTPUT_DIR / "sample_summary.csv").exists():
    summary_df = pd.read_csv(OUTPUT_DIR / "sample_summary.csv")
else:
    raise FileNotFoundError("Missing sample_summary.csv. Run the sample-building cell first.")
display(summary_df)

if "GENERATED_EVENTS_BY_BENCHMARK" in globals():
    display(pd.DataFrame([
        {"benchmark": name, "generated_events_requested": GENERATED_EVENTS_BY_BENCHMARK[name]}
        for name in BENCHMARK_NAMES
    ]))
