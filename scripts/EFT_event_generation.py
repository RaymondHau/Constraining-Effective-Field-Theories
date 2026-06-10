#!/usr/bin/env python
# coding: utf-8

# # EFT event generation
# 
# This notebook only controls MadGraph process creation and benchmark event generation.
# 
# Run it when you want fresh LHE files. Smearing, analysis cuts, plotting, splitting, and MadMiner-style sample preparation now live in `EFT_prepare_madminer_style_samples.ipynb`.
# 

# ## 1. Imports

# In[1]:


from __future__ import annotations

import copy
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Set default plotting parameters
plt.rcParams.update({"figure.figsize": (7, 4), "axes.grid": False})

# Select hardware device for PyTorch computations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


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
# 
# This is the control panel. For a different process, change the model, process string, EFT block/operator keys, benchmark points, generation settings, observables, and cuts here. The rest of the notebook should not need structural edits.
# 

# In[2]:


# -------------------------
# MadGraph and file layout
# -------------------------
# Path to MadGraph 5 installation and binary
MG5_DIR = Path(os.environ.get("MG5_DIR", "external/MG5_aMC_v3_7_1")).expanduser()
if not MG5_DIR.is_absolute():
    MG5_DIR = (Path.cwd() / MG5_DIR).resolve()
MG5_BIN = MG5_DIR / "bin" / "mg5_aMC"

# Directory structure for output files
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
COMMAND_DIR = LOCAL_WORKSPACE_DIR / "mg5_commands"
TABLE_DIR = STORAGE_WORKSPACE_DIR / "tables"
PROCESS_DIR = LOCAL_WORKSPACE_DIR / "processes" / "PROC_EWdim6_VBF_HAA"
GENERATED_LHE_ARCHIVE_DIR = STORAGE_WORKSPACE_DIR / "generated_lhe_archive"
EVENT_GENERATION_LOG_DIR = STORAGE_WORKSPACE_DIR / "event_generation_logs"
ARCHIVE_GENERATED_LHE_TO_STORAGE = os.environ.get("EFT_ARCHIVE_LHE_TO_STORAGE", "1") != "0"
LHE_FILENAME = "unweighted_events.lhe.gz"

_PATH_CONFIG = config_section("paths")
if _PATH_CONFIG:
    if "mg5_dir" in _PATH_CONFIG:
        MG5_DIR = Path(_PATH_CONFIG["mg5_dir"]).expanduser()
        if not MG5_DIR.is_absolute():
            MG5_DIR = (Path.cwd() / MG5_DIR).resolve()
        MG5_BIN = MG5_DIR / "bin" / "mg5_aMC"
    if "local_workspace" in _PATH_CONFIG:
        LOCAL_WORKSPACE_DIR = Path(_PATH_CONFIG["local_workspace"]).expanduser()
    if "storage_workspace" in _PATH_CONFIG:
        STORAGE_WORKSPACE_DIR = Path(_PATH_CONFIG["storage_workspace"]).expanduser()
        STORAGE_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR = LOCAL_WORKSPACE_DIR
    COMMAND_DIR = LOCAL_WORKSPACE_DIR / _PATH_CONFIG.get("command_subdir", "mg5_commands")
    TABLE_DIR = STORAGE_WORKSPACE_DIR / _PATH_CONFIG.get("table_subdir", "tables")
    PROCESS_DIR = LOCAL_WORKSPACE_DIR / _PATH_CONFIG.get("process_subdir", "processes/PROC_EWdim6_VBF_HAA")
    GENERATED_LHE_ARCHIVE_DIR = STORAGE_WORKSPACE_DIR / _PATH_CONFIG.get("lhe_archive_subdir", "generated_lhe_archive")
    EVENT_GENERATION_LOG_DIR = STORAGE_WORKSPACE_DIR / _PATH_CONFIG.get("event_log_subdir", "event_generation_logs")
    ARCHIVE_GENERATED_LHE_TO_STORAGE = bool(_PATH_CONFIG.get("archive_generated_lhe", ARCHIVE_GENERATED_LHE_TO_STORAGE))
    LHE_FILENAME = _PATH_CONFIG.get("lhe_filename", LHE_FILENAME)

# -------------------------
# Physics process
# -------------------------
# UFO model and process definition for VBF Higgs production decaying to two photons
MG_MODEL = "EWdim6-full"
MG_PROCESS = "u d > u d h / a z QCD=0 QED=99 NP=2, h > a a QCD=0 QED=99 NP=0"

# EFT Operator configuration and scaling
EFT_BLOCK = "dim6"
EFT_OPERATORS = ["CWL2", "CPWL2"]
EFT_OPERATOR_KEYS = {"CWL2": "2", "CPWL2": "5"}
PARAM_CARD_VALUE = lambda theta_value: 16.52 * theta_value

# Benchmark points for morphing basis
BENCHMARK_POINTS = {
    "sm":     {"CWL2": 0.0,   "CPWL2": 0.0},
    "w":      {"CWL2": 15.2,  "CPWL2": 0.1},
    "neg_w":  {"CWL2": -15.4, "CPWL2": 0.2},
    "ww":     {"CWL2": 0.3,   "CPWL2": 15.1},
    "neg_ww": {"CWL2": 0.4,   "CPWL2": -15.3},
    "w_ww":   {"CWL2": 16.88, "CPWL2": 14.95},
}

REFERENCE_BENCHMARK = "sm"
REFERENCE_THETA = BENCHMARK_POINTS[REFERENCE_BENCHMARK]

# Allowed parameter ranges for scans
THETA_RANGES = {"CWL2": (-16.0, 17.0), "CPWL2": (-16.0, 16.0)}

_PHYSICS_CONFIG = config_section("physics")
if _PHYSICS_CONFIG:
    MG_MODEL = _PHYSICS_CONFIG.get("mg_model", MG_MODEL)
    MG_PROCESS = _PHYSICS_CONFIG.get("mg_process", MG_PROCESS)
    EFT_BLOCK = _PHYSICS_CONFIG.get("eft_block", EFT_BLOCK)
    EFT_OPERATORS = list(_PHYSICS_CONFIG.get("eft_operators", EFT_OPERATORS))
    EFT_OPERATOR_KEYS = dict(_PHYSICS_CONFIG.get("eft_operator_keys", EFT_OPERATOR_KEYS))
    PARAM_CARD_SCALE = float(_PHYSICS_CONFIG.get("param_card_scale", 16.52))
    PARAM_CARD_VALUE = lambda theta_value, scale=PARAM_CARD_SCALE: scale * theta_value
    BENCHMARK_POINTS = {
        name: {op: float(value) for op, value in theta.items()}
        for name, theta in _PHYSICS_CONFIG.get("benchmark_points", BENCHMARK_POINTS).items()
    }
    REFERENCE_BENCHMARK = _PHYSICS_CONFIG.get("reference_benchmark", REFERENCE_BENCHMARK)
    REFERENCE_THETA = BENCHMARK_POINTS[REFERENCE_BENCHMARK]
    THETA_RANGES = {
        name: tuple(bounds)
        for name, bounds in _PHYSICS_CONFIG.get("theta_ranges", THETA_RANGES).items()
    }

# -------------------------
# Generated-event budgets
# -------------------------
# Paper convention: sample sizes are generated-event budgets, not the number of
# morphed/augmented rows later emitted for neural-network training.
#
# Modes:
#   paper_parameterized_morphing: 10M generated events total, split as
#       reference + morphing-basis samples. For this 2D quadratic basis with
#       N non-reference generated hypotheses, each non-reference point receives
#       10M / (2N) events and the reference receives N times that amount.
#   quick_test: the old small workflow, useful for smoke tests.
EVENT_BUDGET_MODE = os.environ.get("EFT_EVENT_BUDGET_MODE", "paper_parameterized_morphing").lower()
PAPER_PARAMETERIZED_GENERATED_EVENTS = int(os.environ.get("EFT_PAPER_PARAMETERIZED_GENERATED_EVENTS", "10000000"))
QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK = int(os.environ.get("EFT_QUICK_TEST_EVENTS_PER_BENCHMARK", "100000"))
MAX_EVENTS_PER_MG_RUN = int(os.environ.get("EFT_MAX_EVENTS_PER_MG_RUN", "1000000"))
RANDOM_SEED = 42              # Base random seed for generation
LHC_BEAM_ENERGY_GEV = 6500.0  # Proton beam energy (6.5 TeV beam / 13 TeV collision energy)

_GENERATION_CONFIG = config_section("generation")
EVENT_BUDGET_MODE = str(_GENERATION_CONFIG.get("event_budget_mode", EVENT_BUDGET_MODE)).lower()
PAPER_PARAMETERIZED_GENERATED_EVENTS = int(_GENERATION_CONFIG.get("paper_parameterized_generated_events", PAPER_PARAMETERIZED_GENERATED_EVENTS))
QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK = int(_GENERATION_CONFIG.get("quick_test_generated_events_per_benchmark", QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK))
MAX_EVENTS_PER_MG_RUN = int(_GENERATION_CONFIG.get("max_events_per_mg_run", MAX_EVENTS_PER_MG_RUN))
RANDOM_SEED = int(_GENERATION_CONFIG.get("random_seed", RANDOM_SEED))
LHC_BEAM_ENERGY_GEV = float(_GENERATION_CONFIG.get("lhc_beam_energy_gev", LHC_BEAM_ENERGY_GEV))

NON_REFERENCE_BENCHMARK_NAMES = [name for name in BENCHMARK_POINTS if name != REFERENCE_BENCHMARK]
if EVENT_BUDGET_MODE == "paper_parameterized_morphing":
    events_per_basis_hypothesis = PAPER_PARAMETERIZED_GENERATED_EVENTS // (2 * len(NON_REFERENCE_BENCHMARK_NAMES))
    GENERATED_EVENTS_BY_BENCHMARK = {
        REFERENCE_BENCHMARK: events_per_basis_hypothesis * len(NON_REFERENCE_BENCHMARK_NAMES),
        **{name: events_per_basis_hypothesis for name in NON_REFERENCE_BENCHMARK_NAMES},
    }
elif EVENT_BUDGET_MODE == "quick_test":
    GENERATED_EVENTS_BY_BENCHMARK = {name: QUICK_TEST_GENERATED_EVENTS_PER_BENCHMARK for name in BENCHMARK_POINTS}
else:
    raise ValueError(f"Unknown EVENT_BUDGET_MODE={EVENT_BUDGET_MODE!r}")

# Backwards-compatible display value only. Per-benchmark generation uses the
# GENERATED_EVENTS_BY_BENCHMARK dictionary below.
NEVENTS = max(GENERATED_EVENTS_BY_BENCHMARK.values())


# ## 3. Cards and MadGraph automation
# 
# Cards are written by passing literal strings to `write_card`. The only automatic card edits below are the repeated changes needed to move between benchmark parameter points.

# In[ ]:


@dataclass(frozen=True)
class Benchmark:
    """Represents an EFT benchmark point with a name and operator values."""
    name: str
    theta: Dict[str, float]


BENCHMARKS = [Benchmark(name, theta) for name, theta in BENCHMARK_POINTS.items()]
BENCHMARK_NAMES = [b.name for b in BENCHMARKS]
WEIGHT_COLUMNS = [f"w_{name}" for name in BENCHMARK_NAMES]


def generated_events_for_benchmark(benchmark_name: str) -> int:
    """Return the generated-event budget assigned to one benchmark."""
    return int(GENERATED_EVENTS_BY_BENCHMARK[benchmark_name])


def benchmark_run_parts(benchmark_name: str) -> List[int]:
    """Split large generated-event budgets into MadGraph-sized run parts."""
    total = generated_events_for_benchmark(benchmark_name)
    if total <= 0:
        raise ValueError(f"Generated-event budget for {benchmark_name} must be positive")
    full_parts, remainder = divmod(total, MAX_EVENTS_PER_MG_RUN)
    parts = [MAX_EVENTS_PER_MG_RUN] * full_parts
    if remainder:
        parts.append(remainder)
    return parts


def archive_lhe_to_storage(run_dir: Path) -> Path:
    """Move bulky LHE output to storage and leave a local symlink for later notebooks."""
    local_lhe = run_dir / LHE_FILENAME
    if not ARCHIVE_GENERATED_LHE_TO_STORAGE or STORAGE_WORKSPACE_DIR == LOCAL_WORKSPACE_DIR:
        return local_lhe
    if local_lhe.is_symlink():
        return local_lhe
    if not local_lhe.exists():
        raise FileNotFoundError(f"Expected generated LHE file is missing: {local_lhe}")
    archive_lhe = GENERATED_LHE_ARCHIVE_DIR / run_dir.name / LHE_FILENAME
    archive_lhe.parent.mkdir(parents=True, exist_ok=True)
    if archive_lhe.exists():
        local_lhe.unlink()
    else:
        shutil.move(str(local_lhe), str(archive_lhe))
    local_lhe.symlink_to(archive_lhe)
    print(f"Archived LHE to {archive_lhe} and linked from {local_lhe}")
    return local_lhe


def write_card(path: Path, contents: str) -> Path:
    """Writes a MadGraph card to the specified path, ensuring parent directories exist."""
    # Create target directories if they do not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write the card configuration to file
    path.write_text(contents, encoding="utf-8")
    return path


def format_command(command: Sequence[str] | str) -> str:
    """Formats a command list or string into a single command line for display."""
    # Return directly if already a string, otherwise join list elements
    if isinstance(command, str):
        return command
    return " ".join(map(str, command))


def run_command(command: Sequence[str] | str, cwd: Path | None = None, label: str | None = None) -> str:
    """Run a shell command while streaming verbose output to a log file.

    MadGraph can print enormous logs during production runs. Keeping that output
    in a notebook cell or in subprocess.PIPE can destabilize the kernel, so this
    function writes stdout/stderr directly to disk and only prints the log path.
    """
    description = label or format_command(command)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", description).strip("_")[:160] or "command"
    EVENT_GENERATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EVENT_GENERATION_LOG_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name}.log"

    print(f"Starting: {description}", flush=True)
    print(f"  log: {log_path}", flush=True)
    start_time = time.monotonic()

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {format_command(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=isinstance(command, str),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        returncode = process.wait()

    elapsed = time.monotonic() - start_time
    if returncode != 0:
        try:
            output_tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        except OSError:
            output_tail = f"Could not read log file {log_path}"
        print(f"Failed: {description} after {elapsed:.1f} s")
        print(output_tail)
        raise RuntimeError(f"Command failed with exit code {returncode}: {description}. See {log_path}")

    print(f"Done: {description} ({elapsed:.1f} s). Log: {log_path}", flush=True)
    return str(log_path)


PROCESS_CARD = f"""
import model {MG_MODEL}
generate {MG_PROCESS}
output {PROCESS_DIR} -f
""".strip() + "\n"


# Explicit run_card.dat template; placeholders are filled before each benchmark run.
RUN_CARD_TEMPLATE = """
#*********************************************************************
#                       MadGraph5_aMC@NLO                            *
#                                                                    *
#                     run_card.dat MadEvent                          *
#                                                                    *
#  This file is used to set the parameters of the run.               *
#                                                                    *
#  Some notation/conventions:                                        *
#                                                                    *
#   Lines starting with a '# ' are info or comments                  *
#                                                                    *
#   mind the format:   value    = variable     ! comment             *
#                                                                    *
#   To display more options, you can type the command:               *
#      update to_full                                                *
#*********************************************************************
#                                                                    
#*********************************************************************
# Tag name for the run (one word)                                    *
#*********************************************************************
  tag_1     = run_tag ! name of the run 
#*********************************************************************
# Number of events and rnd seed                                      *
# Warning: Do not generate more than 1M events in a single run       *
#*********************************************************************
  __NEVENTS__ = nevents ! Number of unweighted events requested
  __SEED__   = iseed   ! rnd seed (0=assigned automatically=default))
#*********************************************************************
# Collider type and energy                                           *
# lpp: 0=No PDF, 1=proton, -1=antiproton,                            *
#                2=elastic photon of proton/ion beam                 *
#             +/-3=PDF of electron/positron beam                     *
#             +/-4=PDF of muon/antimuon beam                         *
#*********************************************************************
     1        = lpp1    ! beam 1 type 
     1        = lpp2    ! beam 2 type
     __BEAM_ENERGY__     = ebeam1  ! beam 1 total energy in GeV
     __BEAM_ENERGY__     = ebeam2  ! beam 2 total energy in GeV
# To see polarised beam options: type "update beam_pol"

#*********************************************************************
# PDF CHOICE: this automatically fixes alpha_s and its evol.         *
# pdlabel: lhapdf=LHAPDF (installation needed) [1412.7420]           *
#          iww=Improved Weizsaecker-Williams Approx.[hep-ph/9310350] *
#          eva=Effective W/Z/A Approx.       [2111.02442, 2502.07878]*
#          edff=EDFF in gamma-UPC            [eq.(11) in 2207.03012] *
#          chff=ChFF in gamma-UPC            [eq.(13) in 2207.03012] *
#          none=No PDF, same as lhapdf with lppx=0                   *
#*********************************************************************
     nn23lo1    = pdlabel1     ! PDF type for beam #1
     nn23lo1    = pdlabel2     ! PDF type for beam #2
     230000    = lhaid     ! if pdlabel=lhapdf, this is the lhapdf number
# To see heavy ion options: type "update ion_pdf"

#*********************************************************************
# Renormalization and factorization scales                           *
#*********************************************************************
 False = fixed_ren_scale  ! if .true. use fixed ren scale
 False = fixed_fac_scale1  ! if .true. use fixed fac scale for beam 1
 False = fixed_fac_scale2  ! if .true. use fixed fac scale for beam 2
 91.188  = scale            ! fixed ren scale
 91.188  = dsqrt_q2fact1    ! fixed fact scale for pdf1
 91.188  = dsqrt_q2fact2    ! fixed fact scale for pdf2
 -1 = dynamical_scale_choice ! Choose one of the preselected dynamical choices
 1.0  = scalefact        ! scale factor for event-by-event scales

#*********************************************************************
# Type and output format
#*********************************************************************
  False     = gridpack  !True = setting up the grid pack
  -1.0 = time_of_flight ! threshold (in mm) below which the invariant livetime is not written (-1 means not written)
  sum =  event_norm       ! average/sum. Normalization of the weight in the LHEF
# To see MLM/CKKW  merging options: type "update MLM" or "update CKKW"

#*********************************************************************
#
#*********************************************************************
# Phase-Space Optimization strategy (basic options)
#*********************************************************************
   0  = nhel          ! using helicities importance sampling or not.
                             ! 0: sum over helicity, 1: importance sampling
   2  = sde_strategy  ! default integration strategy (hep-ph/2021.00773)
                             ! 1 is old strategy (using amp square)
			     ! 2 is new strategy (using only the denominator)
# To see advanced option for Phase-Space optimization: type "update psoptim"			     
#*********************************************************************
# Customization (custom cuts/scale/bias/...)                         *
# list of files containing fortran function that overwrite default   *
#*********************************************************************
  = custom_fcts ! List of files containing user hook function
#*******************************                                                 
# Parton level cuts definition *
#*******************************
  0.0  = dsqrt_shat ! minimal shat for full process
  -1  = dsqrt_shatmax ! maximum shat for full process
#                                                                    
#
#*********************************************************************
# BW cutoff (M+/-bwcutoff*Gamma) ! Define on/off-shell for "$" and decay  
#*********************************************************************
  15.0  = bwcutoff      ! (M+/-bwcutoff*Gamma)
 #*********************************************************************
 # Apply pt/E/eta/dr/mij/kt_durham cuts on decay products or not
 # (note that etmiss/ptll/ptheavy/ht/sorted cuts always apply)
 #*********************************************************************
   False  = cut_decays    ! Cut decay products 
#*********************************************************************
# Standard Cuts                                                      *
#*********************************************************************
# Minimum and maximum pt's (for max, -1 means no cut)                *
#*********************************************************************
 20.0  = ptj       ! minimum pt for the jets 
 10.0  = pta       ! minimum pt for the photons 
 -1.0  = ptjmax    ! maximum pt for the jets
 -1.0  = ptamax    ! maximum pt for the photons
 {} = pt_min_pdg ! pt cut for other particles (use pdg code). Applied on particle and anti-particle
 {}	= pt_max_pdg ! pt cut for other particles (syntax e.g. {6: 100, 25: 50}) 
#
# For display option for energy cut in the partonic center of mass frame type 'update ecut'
#
#*********************************************************************
# Maximum and minimum absolute rapidity (for max, -1 means no cut)   *
#*********************************************************************
 5.0 = etaj    ! max rap for the jets 
 2.5  = etaa    ! max rap for the photons 
 0.0  = etajmin ! min rap for the jets
 0.0  = etaamin ! min rap for the photons
 {} = eta_min_pdg ! rap cut for other particles (use pdg code). Applied on particle and anti-particle
 {} = eta_max_pdg ! rap cut for other particles (syntax e.g. {6: 2.5, 23: 5})
#*********************************************************************
# Minimum and maximum DeltaR distance                                *
#*********************************************************************
 0.4 = drjj    ! min distance between jets 
 0.4 = draa    ! min distance between gammas 
 0.4 = draj    ! min distance between gamma and jet 
 -1.0  = drjjmax ! max distance between jets
 -1.0  = draamax ! max distance between gammas
 -1.0  = drajmax ! max distance between gamma and jet
#*********************************************************************
# Minimum and maximum invariant mass for pairs                       *
#*********************************************************************
 0.0   = mmjj    ! min invariant mass of a jet pair 
 0.0   = mmaa    ! min invariant mass of gamma gamma pair
 -1.0  = mmjjmax ! max invariant mass of a jet pair
 -1.0  = mmaamax ! max invariant mass of gamma gamma pair
 {} = mxx_min_pdg ! min invariant mass of a pair of particles X/X~ (e.g. {6:250})
 {'default': False} = mxx_only_part_antipart ! if True the invariant mass is applied only 
                       ! to pairs of particle/antiparticle and not to pairs of the same pdg codes.  
#*********************************************************************
# Inclusive cuts                                                     *
#*********************************************************************
 0.0  = xptj ! minimum pt for at least one jet  
 0.0  = xpta ! minimum pt for at least one photon 
 #*********************************************************************
 # Control the pt's of the jets sorted by pt                          *
 #*********************************************************************
 0.0   = ptj1min ! minimum pt for the leading jet in pt
 0.0   = ptj2min ! minimum pt for the second jet in pt
 -1.0  = ptj1max ! maximum pt for the leading jet in pt 
 -1.0  = ptj2max ! maximum pt for the second jet in pt
 0   = cutuse  ! reject event if fails any (0) / all (1) jet pt cuts
 #*********************************************************************
 # Control the Ht(k)=Sum of k leading jets                            *
 #*********************************************************************
 0.0   = htjmin ! minimum jet HT=Sum(jet pt)
 -1.0  = htjmax ! maximum jet HT=Sum(jet pt)
 0.0   = ihtmin  !inclusive Ht for all partons (including b)
 -1.0  = ihtmax  !inclusive Ht for all partons (including b)
 #***********************************************************************
 # Photon-isolation cuts, according to hep-ph/9801442                   *
 # When ptgmin=0, all the other parameters are ignored                  *
 # When ptgmin>0, pta and draj are not going to be used                 *
 #***********************************************************************
  0.0 = ptgmin ! Min photon transverse momentum
  0.4 = R0gamma ! Radius of isolation code
  1.0 = xn ! n parameter of eq.(3.4) in hep-ph/9801442
  1.0 = epsgamma ! epsilon_gamma parameter of eq.(3.4) in hep-ph/9801442
  True = isoEM ! isolate photons from EM energy (photons and leptons)
 #*********************************************************************
 # WBF cuts                                                           *
 #*********************************************************************
 0.0   = xetamin ! minimum rapidity for two jets in the WBF case  
 0.0   = deltaeta ! minimum rapidity for two jets in the WBF case 
#*********************************************************************
# maximal pdg code for quark to be considered as a light jet         *
# (otherwise b cuts are applied)                                     *
#*********************************************************************
 4 = maxjetflavor    ! Maximum jet pdg code
#*********************************************************************
#
#*********************************************************************
# Store info for systematics studies                                 *
# WARNING: Do not use for interference type of computation           *
#*********************************************************************
   True  = use_syst      ! Enable systematics studies
#
systematics = systematics_program ! none, systematics [python], SysCalc [depreceted, C++]
['--mur=0.5,1,2', '--muf=0.5,1,2', '--pdf=errorset'] = systematics_arguments ! see: https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/Systematics#Systematicspythonmodule
"""

# Static param_card.dat content after the notebook-written dim6 block.
PARAM_CARD_STATIC_TAIL = """
###################################
## INFORMATION FOR MASS
###################################
Block mass 
    4 1.270000e+00 # MC 
    5 4.700000e+00 # MB 
    6 1.720000e+02 # MT 
   13 1.056600e-01 # MM 
   15 1.777000e+00 # MTA 
   23 9.118760e+01 # MZ 
   25 1.250000e+02 # MH 
  9000006 1.250000e+02 # MP 
## Dependent parameters, given by model restrictions.
## Those values should be edited following the 
## analytical expression. MG5 ignores those values 
## but they are important for interfacing the output of MG5
## to external program such as Pythia.
  1 0.000000e+00 # d : 0.0 
  2 0.000000e+00 # u : 0.0 
  3 0.000000e+00 # s : 0.0 
  11 0.000000e+00 # e- : 0.0 
  12 0.000000e+00 # ve : 0.0 
  14 0.000000e+00 # vm : 0.0 
  16 0.000000e+00 # vt : 0.0 
  21 0.000000e+00 # g : 0.0 
  22 0.000000e+00 # a : 0.0 
  24 7.982436e+01 # w+ : cmath.sqrt(MZ__exp__2/2. + cmath.sqrt(MZ__exp__4/4. - (aEW*cmath.pi*MZ__exp__2)/(Gf*sqrt__2))) 

###################################
## INFORMATION FOR SMINPUTS
###################################
Block sminputs 
    1 1.279000e+02 # aEWM1 
    2 1.166370e-05 # Gf 
    3 1.184000e-01 # aS (Note: this Parameter is not used if you use a PDF set) 

###################################
## INFORMATION FOR WOLFENSTEIN
###################################
Block wolfenstein 
    1 0.000000e+00 # lamWS 
    2 0.000000e+00 # AWS 
    3 0.000000e+00 # rhoWS 
    4 0.000000e+00 # etaWS 

###################################
## INFORMATION FOR YUKAWA
###################################
Block yukawa 
    4 1.270000e+00 # ymc 
    5 4.700000e+00 # ymb 
    6 1.720000e+02 # ymt 
   13 1.056600e-01 # ymm 
   15 1.777000e+00 # ymtau 

###################################
## INFORMATION FOR DECAY
###################################
DECAY   6 1.508336e+00 # WT 
DECAY  15 2.270000e-12 # WTau 
DECAY  23 2.495200e+00 # WZ 
DECAY  24 2.085000e+00 # WW 
DECAY  25 6.382339e-03 # WH 
DECAY 9000006 6.382339e-03 # WH1 
## Dependent parameters, given by model restrictions.
## Those values should be edited following the 
## analytical expression. MG5 ignores those values 
## but they are important for interfacing the output of MG5
## to external program such as Pythia.
DECAY  1 0.000000e+00 # d : 0.0 
DECAY  2 0.000000e+00 # u : 0.0 
DECAY  3 0.000000e+00 # s : 0.0 
DECAY  4 0.000000e+00 # c : 0.0 
DECAY  5 0.000000e+00 # b : 0.0 
DECAY  11 0.000000e+00 # e- : 0.0 
DECAY  12 0.000000e+00 # ve : 0.0 
DECAY  13 0.000000e+00 # mu- : 0.0 
DECAY  14 0.000000e+00 # vm : 0.0 
DECAY  16 0.000000e+00 # vt : 0.0 
DECAY  21 0.000000e+00 # g : 0.0 
DECAY  22 0.000000e+00 # a : 0.0 
#===========================================================
# QUANTUM NUMBERS OF NEW STATE(S) (NON SM PDG CODE)
#===========================================================

Block QNUMBERS 9000006  # h1 
        1 0  # 3 times electric charge
        2 1  # number of spin states (2S+1)
        3 1  # colour rep (1: singlet, 3: triplet, 8: octet)
        4 0  # Particle/Antiparticle distinction (0=own anti)
"""


def make_process_dir() -> None:
    """Creates a fresh MadGraph process directory by removing any existing one and running MG5."""
    # Remove old process directory if it exists to avoid conflicts
    if PROCESS_DIR.exists():
        shutil.rmtree(PROCESS_DIR)

    # Write MG5 command file and run the process generator
    card_path = write_card(COMMAND_DIR / "make_process.mg5", PROCESS_CARD)
    run_command([str(MG5_BIN), str(card_path)], label="Create MadGraph process")


def configure_run_card(seed: int, nevents: int) -> Path:
    """Generates the run_card.dat for MadGraph, dynamically substituting configuration parameters."""
    # Populate the template with runtime parameters
    text = RUN_CARD_TEMPLATE.replace("__NEVENTS__", str(nevents))
    text = text.replace("__SEED__", str(seed))
    text = text.replace("__BEAM_ENERGY__", str(LHC_BEAM_ENERGY_GEV))
    return write_card(PROCESS_DIR / "Cards" / "run_card.dat", text)


def lha_assignments(theta: Mapping[str, float]) -> Dict[str, float]:
    """Translates high-level EFT parameter values to their scaled param_card counterparts."""
    # Scale each active operator according to its parameter card mapping
    return {
        EFT_OPERATOR_KEYS[name]: PARAM_CARD_VALUE(theta.get(name, 0.0))
        for name in EFT_OPERATORS
    }


def make_dim6_block(theta: Mapping[str, float]) -> str:
    """Formats the LHA 'dim6' block for the param_card, ensuring unselected parameters default to zero."""
    # Retrieve scaled operator values and initialize all 8 dim6 fields to 0.0
    assignments = lha_assignments(theta)
    values = {str(index): 0.0 for index in range(1, 9)}
    values.update(assignments)

    # Format the block according to Les Houches Accord conventions
    return f"""###################################
## INFORMATION FOR DIM6
###################################
Block dim6
    1 {values["1"]: .12e} # CWWWL2
    2 {values["2"]: .12e} # CWL2
    3 {values["3"]: .12e} # CBL2
    4 {values["4"]: .12e} # CPWWWL2
    5 {values["5"]: .12e} # CPWL2
    6 {values["6"]: .12e} # CphidL2
    7 {values["7"]: .12e} # CphiWL2
    8 {values["8"]: .12e} # CphiBL2

"""


def make_param_card(theta: Mapping[str, float]) -> str:
    """Assembles the complete param_card.dat text for a given EFT benchmark point."""
    header = """######################################################################
## PARAM_CARD EXPLICITLY WRITTEN BY THE NOTEBOOK                   ####
######################################################################
##                                                                  ##
## The dim6 block is filled from BENCHMARK_POINTS.                  ##
## Unselected EFT coefficients are always written as zero.          ##
##                                                                  ##
######################################################################

"""
    # Combine the documentation header, dim6 settings, and trailing SM model details
    return header + make_dim6_block(theta) + PARAM_CARD_STATIC_TAIL


def write_param_card(theta: Mapping[str, float]) -> Path:
    """Overwrites the active param_card.dat in the MadGraph Cards directory."""
    # Output the formatted parameter card text to file
    return write_card(PROCESS_DIR / "Cards" / "param_card.dat", make_param_card(theta))


def make_reweight_card() -> Path:
    """Generates the reweight_card.dat instructing MadGraph to evaluate weights at all benchmarks."""
    lines = ["# Native MadGraph reweight points used for morphing and joint targets"]

    # Append setting block for each configured benchmark point
    for benchmark in BENCHMARKS:
        lines.append(f"launch --rwgt_name={benchmark.name}")
        for key, value in lha_assignments(benchmark.theta).items():
            lines.append(f"  set {EFT_BLOCK} {key} {value:.12e}")

    # Save reweight configuration file
    return write_card(PROCESS_DIR / "Cards" / "reweight_card.dat", "\n".join(lines) + "\n")


def generate_benchmark(benchmark: Benchmark, index: int) -> List[Path]:
    """Run one benchmark, possibly split into multiple generated-event parts."""
    run_dirs = []
    parts = benchmark_run_parts(benchmark.name)
    for part_index, part_nevents in enumerate(parts, start=1):
        suffix = f"_part{part_index:02d}" if len(parts) > 1 else ""
        run_name = f"basis_{index:02d}_{benchmark.name}{suffix}"
        run_dir = PROCESS_DIR / "Events" / run_name

        # Configure input cards before running generation. The generated-event
        # budget lives here; later morphing/augmentation does not redefine it.
        configure_run_card(seed=RANDOM_SEED + 1000 * index + part_index, nevents=part_nevents)
        write_param_card(benchmark.theta)
        make_reweight_card()

        generate_events = PROCESS_DIR / "bin" / "generate_events"
        command = f'yes 0 | "{generate_events}" "{run_name}" -f'
        label = (
            f"Generate {benchmark.name} part {part_index}/{len(parts)} "
            f"({part_nevents:,} generated events)"
        )
        run_command(["bash", "-lc", command], cwd=PROCESS_DIR, label=label)
        archive_lhe_to_storage(run_dir)
        run_dirs.append(run_dir)
    return run_dirs


def generate_all_benchmarks() -> Dict[str, List[Path]]:
    """Generate MC samples using explicit generated-event budgets."""
    print("Generated-event budget mode:", EVENT_BUDGET_MODE)
    print("Generated-event budgets:", GENERATED_EVENTS_BY_BENCHMARK)
    make_process_dir()

    run_dirs = {}
    for i, benchmark in enumerate(BENCHMARKS):
        run_dirs[benchmark.name] = generate_benchmark(benchmark, i)

    print("MadGraph generation complete.")
    return run_dirs


# In[4]:


RUN_DIRS = generate_all_benchmarks()
RUN_DIRS


# ## Next step
# 
# After MadGraph finishes, run `EFT_prepare_madminer_style_samples.ipynb` to parse the generated LHE files, apply smearing and cuts, make diagnostics, and build neural-training sample tables.
