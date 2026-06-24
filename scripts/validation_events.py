#!/usr/bin/env python3
"""Generate validation datasets with the same MadGraph flow as event generation.

The validation pipeline deliberately mirrors the known-good
``EFT_event_generation.py`` pattern:

1. create a MadGraph process directory once;
2. overwrite ``Cards/run_card.dat`` and ``Cards/param_card.dat`` for each point;
3. run ``bin/generate_events`` directly;
4. parse the produced LHE and apply smearing/cuts here;
5. write ``features.npy`` for downstream likelihood scans.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESS_NAME = os.environ.get("EFT_PROCESS", "WBF")
PROCESS_CONFIG_DIR = PROJECT_DIR / "configs" / PROCESS_NAME
DEFAULT_CONFIG_PATH = PROCESS_CONFIG_DIR / "validation_events_config.json"
DEFAULT_PREPARATION_CONFIG_PATH = PROCESS_CONFIG_DIR / "sample_preparation.json"

FINAL_STATE_STATUS = 1
JET_PDGS = {1, 2, 3, 4, 5, 21}
PHOTON_PDGS = {22}
LEPTON_PDGS = {11, 13}
INVISIBLE_PDGS = {12, 14, 16}

DIM6_LABELS = {
    1: "CWWWL2",
    2: "CWL2",
    3: "CBL2",
    4: "CPWWWL2",
    5: "CPWL2",
    6: "CphidL2",
    7: "CphiWL2",
    8: "CphiBL2",
}


@dataclass(frozen=True)
class ThetaPoint:
    """A two-dimensional EFT point."""

    c1: float
    c2: float

    @classmethod
    def from_value(cls, value: Any) -> "ThetaPoint":
        if isinstance(value, Mapping):
            return cls(float(value["c1"]), float(value["c2"]))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(float(value[0]), float(value[1]))
        raise ValueError(f"Invalid theta point {value!r}; expected [c1, c2] or c1/c2 dict.")

    @property
    def tag(self) -> str:
        return f"c1_{self.c1:+.6g}_c2_{self.c2:+.6g}".replace("+", "p").replace("-", "m").replace(".", "p")


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    config["_config_dir"] = str(config_path.parent)
    config["_project_dir"] = str(PROJECT_DIR)
    return config


def resolve_config_path(config: Mapping[str, Any], value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_project_dir"]) / path


def portable_project_path(config: Mapping[str, Any], value: str | os.PathLike[str]) -> str:
    """Prefer repository-relative paths in metadata copied between machines."""

    path = Path(value).resolve()
    project_dir = Path(config["_project_dir"]).resolve()
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)


def resolve_executable(config: Mapping[str, Any], value: str | os.PathLike[str]) -> str:
    text = str(value)
    if os.sep not in text and "/" not in text:
        return text
    return str(resolve_config_path(config, text))


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def format_command(command: Sequence[str] | str) -> str:
    if isinstance(command, str):
        return command
    return " ".join(map(str, command))


def run_command(
    command: Sequence[str] | str,
    log_path: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print(f"Starting: {format_command(command)}", flush=True)
    print(f"  log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {format_command(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            shell=isinstance(command, str),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"Command failed with exit code {return_code}: {format_command(command)}\n{tail}")
    print(f"Done in {time.monotonic() - start:.1f} s", flush=True)


def process_card_text(config: Mapping[str, Any], process_dir: Path) -> str:
    lines: list[str] = []
    if config.get("import_model"):
        lines.append(f"import model {config['import_model']}")
    lines.append(f"generate {config['process']}")
    for addition in config.get("add_processes", []):
        lines.append(f"add process {addition}")
    lines.append(f"output {process_dir} -f")
    return "\n".join(lines) + "\n"


def make_process_dir(config: Mapping[str, Any], output_root: Path) -> tuple[Path, Path]:
    process_dir = output_root / str(config.get("process_subdir", "process"))
    command_dir = output_root / str(config.get("command_subdir", "mg5_commands"))
    log_dir = resolve_config_path(config, config.get("log_dir", "logs/madgraph/validation_events"))
    if process_dir.exists() and bool(config.get("recreate_process", True)):
        shutil.rmtree(process_dir)
    if process_dir.exists():
        return process_dir, log_dir

    script_path = write_text(command_dir / "make_validation_process.mg5", process_card_text(config, process_dir))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in config.get("environment", {}).items()})
    madgraph = resolve_executable(config, config.get("madgraph_executable", "mg5_aMC"))
    run_command([madgraph, str(script_path)], log_dir / "make_validation_process.log", cwd=output_root, env=env)
    return process_dir, log_dir


def render_template(value: Any, theta: ThetaPoint, coefficient_names: Sequence[str], scale: float) -> str:
    if not isinstance(value, str):
        return str(value)
    if len(coefficient_names) != 2:
        raise ValueError("coefficient_names must contain exactly two names.")
    values = {
        "c1": theta.c1,
        "c2": theta.c2,
        "c1_param": scale * theta.c1,
        "c2_param": scale * theta.c2,
        coefficient_names[0]: theta.c1,
        coefficient_names[1]: theta.c2,
        f"{coefficient_names[0]}_param": scale * theta.c1,
        f"{coefficient_names[1]}_param": scale * theta.c2,
    }
    return value.format(**values)


def set_run_card_value(text: str, variable: str, value: Any) -> str:
    pattern = re.compile(rf"^(\s*)(\S+)(\s*=\s*{re.escape(variable)}\b.*)$", re.MULTILINE)
    replacement = rf"\g<1>{value}\g<3>"
    text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        raise KeyError(f"Could not find run_card variable {variable!r}.")
    return text


def configure_run_card(config: Mapping[str, Any], process_dir: Path, seed: int, nevents: int) -> Path:
    run_card = process_dir / "Cards" / "run_card.dat"
    text = run_card.read_text(encoding="utf-8", errors="replace")
    edits = {
        "nevents": int(nevents),
        "iseed": int(seed),
        "event_norm": "sum",
        **config.get("run_card_settings", {}),
    }
    for variable, value in edits.items():
        text = set_run_card_value(text, str(variable), value)
    return write_text(run_card, text)


def dim6_values(config: Mapping[str, Any], theta: ThetaPoint) -> dict[int, float]:
    values = {index: 0.0 for index in range(1, 9)}
    coefficient_names = list(config.get("coefficient_names", ["c1", "c2"]))
    scale = float(config.get("param_card_scale", 1.0))
    settings = config.get("param_card_settings", {})
    if settings:
        for key, raw_value in settings.items():
            parts = str(key).split()
            if len(parts) == 2 and parts[0].lower() == "dim6":
                values[int(parts[1])] = float(render_template(raw_value, theta, coefficient_names, scale))
        return values

    operator_keys = config.get("eft_operator_keys", {coefficient_names[0]: "2", coefficient_names[1]: "5"})
    theta_by_name = {coefficient_names[0]: theta.c1, coefficient_names[1]: theta.c2}
    for name in coefficient_names:
        values[int(operator_keys[name])] = scale * theta_by_name[name]
    return values


def make_dim6_block(config: Mapping[str, Any], theta: ThetaPoint) -> str:
    values = dim6_values(config, theta)
    lines = [
        "###################################",
        "## INFORMATION FOR DIM6",
        "###################################",
        "Block dim6",
    ]
    for index in range(1, 9):
        lines.append(f"    {index} {values[index]: .12e} # {DIM6_LABELS.get(index, f'dim6_{index}')}")
    return "\n".join(lines) + "\n\n"


def replace_lha_block(text: str, block_name: str, replacement: str) -> str:
    pattern = re.compile(
        rf"(?ims)^\s*block\s+{re.escape(block_name)}\b.*?(?=^\s*(?:block\b|decay\b)|\Z)"
    )
    text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        raise KeyError(f"Could not find LHA block {block_name!r} in param_card.")
    return text


def configure_param_card(config: Mapping[str, Any], process_dir: Path, theta: ThetaPoint) -> Path:
    param_card = process_dir / "Cards" / "param_card.dat"
    text = param_card.read_text(encoding="utf-8", errors="replace")
    text = replace_lha_block(text, "dim6", make_dim6_block(config, theta))
    return write_text(param_card, text)


def find_event_file(run_dir: Path, preferred_name: str = "unweighted_events.lhe.gz") -> Path:
    preferred = run_dir / preferred_name
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob("*unweighted_events*.lhe*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No unweighted LHE file found under {run_dir}.")
    return candidates[0]


def copy_event_file(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def open_text_maybe_gzip(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def parse_event_block(lines: list[str]) -> dict[str, Any]:
    clean = [line for line in lines if line.strip()]
    header = clean[0].split()
    n_particles = int(header[0])
    particles = []
    for row in clean[1 : 1 + n_particles]:
        values = row.split()
        particles.append(
            {
                "pdg_id": int(values[0]),
                "status": int(values[1]),
                "px": float(values[6]),
                "py": float(values[7]),
                "pz": float(values[8]),
                "e": float(values[9]),
                "mass": float(values[10]),
            }
        )
    return {"particles": particles}


def parse_lhe_events(path: str | Path):
    inside = False
    lines: list[str] = []
    with open_text_maybe_gzip(path) as handle:
        for line in handle:
            if "<event" in line:
                inside = True
                lines = []
                continue
            if "</event>" in line:
                yield parse_event_block(lines)
                inside = False
                continue
            if inside:
                lines.append(line.rstrip("\n"))


def pt(px: float, py: float) -> float:
    return math.hypot(px, py)


def eta(px: float, py: float, pz: float) -> float:
    p = math.sqrt(px * px + py * py + pz * pz)
    return 0.5 * math.log((p + pz + 1.0e-12) / (p - pz + 1.0e-12))


def phi(px: float, py: float) -> float:
    return math.atan2(py, px)


def delta_phi(phi_a: float, phi_b: float) -> float:
    return (phi_a - phi_b + math.pi) % (2.0 * math.pi) - math.pi


def delta_r(obj_a: Mapping[str, float], obj_b: Mapping[str, float]) -> float:
    return math.hypot(obj_a["eta"] - obj_b["eta"], delta_phi(obj_a["phi"], obj_b["phi"]))


def invariant_mass(objects: Sequence[Mapping[str, float]]) -> float:
    energy = sum(p["e"] for p in objects)
    px = sum(p["px"] for p in objects)
    py = sum(p["py"] for p in objects)
    pz = sum(p["pz"] for p in objects)
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def combined_kinematics(objects: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not objects:
        return {"e": np.nan, "pt": np.nan, "eta": np.nan, "phi": np.nan, "mass": np.nan}
    energy = sum(p["e"] for p in objects)
    px = sum(p["px"] for p in objects)
    py = sum(p["py"] for p in objects)
    pz = sum(p["pz"] for p in objects)
    return {"e": energy, "pt": pt(px, py), "eta": eta(px, py, pz), "phi": phi(px, py), "mass": invariant_mass(objects)}


def particle_kinematics(particle: Mapping[str, float]) -> dict[str, float]:
    return {"pt": pt(particle["px"], particle["py"]), "eta": eta(particle["px"], particle["py"], particle["pz"]), "phi": phi(particle["px"], particle["py"])}


def momentum_abs(particle: Mapping[str, float]) -> float:
    return math.sqrt(particle["px"] ** 2 + particle["py"] ** 2 + particle["pz"] ** 2)


def update_energy_consistency(particle: dict[str, float]) -> None:
    mass = max(float(particle["mass"]), 0.0)
    momentum = momentum_abs(particle)
    particle["e"] = max(float(particle["e"]), math.sqrt(momentum * momentum + mass * mass))


def set_energy(particle: dict[str, float], value: float) -> None:
    mass = max(float(particle["mass"]), 0.0)
    current = momentum_abs(particle)
    target_energy = max(float(value), mass)
    target_momentum = math.sqrt(max(target_energy * target_energy - mass * mass, 0.0))
    particle["e"] = target_energy
    if current > 0.0:
        scale = target_momentum / current
        particle["px"] *= scale
        particle["py"] *= scale
        particle["pz"] *= scale


def set_eta(particle: dict[str, float], value: float) -> None:
    kin = particle_kinematics(particle)
    particle["px"] = kin["pt"] * math.cos(kin["phi"])
    particle["py"] = kin["pt"] * math.sin(kin["phi"])
    particle["pz"] = kin["pt"] * math.sinh(float(value))
    update_energy_consistency(particle)


def set_phi(particle: dict[str, float], value: float) -> None:
    kin = particle_kinematics(particle)
    particle["px"] = kin["pt"] * math.cos(float(value))
    particle["py"] = kin["pt"] * math.sin(float(value))
    update_energy_consistency(particle)


def active_smearing_rules(smearing: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    return [(str(feature), float(absolute or 0.0), float(relative or 0.0)) for feature, absolute, relative in smearing.get("rules", [])]


def smear_particle(particle: Mapping[str, float], rules: list[tuple[str, float, float]], rng: np.random.Generator) -> dict[str, float]:
    smeared = dict(particle)
    if particle["status"] != FINAL_STATE_STATUS:
        return smeared
    for feature, absolute, relative in rules:
        if feature == "energy_resolution":
            value = float(smeared["e"])
            sigma = absolute + relative * abs(value)
            set_energy(smeared, value if sigma <= 0.0 else float(rng.normal(value, sigma)))
        elif feature == "eta_resolution":
            value = particle_kinematics(smeared)["eta"]
            sigma = absolute + relative * abs(value)
            set_eta(smeared, value if sigma <= 0.0 else float(rng.normal(value, sigma)))
        elif feature == "phi_resolution":
            value = particle_kinematics(smeared)["phi"]
            sigma = absolute + relative * abs(value)
            set_phi(smeared, value if sigma <= 0.0 else float(rng.normal(value, sigma)))
        else:
            raise KeyError(f"Unknown smearing feature {feature!r}.")
    return smeared


def event_objects(particles: list[Mapping[str, float]]) -> dict[str, Any]:
    final = [p for p in particles if p["status"] == FINAL_STATE_STATUS]
    jets = [dict(p, **particle_kinematics(p)) for p in final if abs(p["pdg_id"]) in JET_PDGS]
    photons = [dict(p, **particle_kinematics(p)) for p in final if abs(p["pdg_id"]) in PHOTON_PDGS]
    leptons = [dict(p, **particle_kinematics(p)) for p in final if abs(p["pdg_id"]) in LEPTON_PDGS]
    return {
        "final": final,
        "jets": sorted(jets, key=lambda p: p["pt"], reverse=True),
        "photons": sorted(photons, key=lambda p: p["pt"], reverse=True),
        "leptons": sorted(leptons, key=lambda p: p["pt"], reverse=True),
        "n_jets": len(jets),
        "n_photons": len(photons),
        "n_leptons": len(leptons),
    }


def extract_observables(particles: list[Mapping[str, float]]) -> tuple[dict[str, float] | None, dict[str, Any]]:
    objects = event_objects(particles)
    jets = objects["jets"]
    photons = objects["photons"]
    leptons = objects["leptons"]
    final = objects["final"]
    if len(jets) < 2:
        return None, objects
    j1, j2 = jets[0], jets[1]
    signed_delta_phi = delta_phi(j1["phi"], j2["phi"]) * (-1.0 + 2.0 * float(j1["eta"] > j2["eta"]))
    delta_eta_jj = j1["eta"] - j2["eta"]
    a1 = photons[0] if len(photons) > 0 else None
    a2 = photons[1] if len(photons) > 1 else None
    diphoton = combined_kinematics(photons[:2]) if len(photons) >= 2 else combined_kinematics([])
    if len(photons) >= 2:
        diphoton_delta_r = delta_r(a1, a2)
        eta_span_jj = abs(j1["eta"] - j2["eta"])
        zeppenfeld_aa = (diphoton["eta"] - 0.5 * (j1["eta"] + j2["eta"])) / eta_span_jj if eta_span_jj > 0.0 else np.nan
    else:
        diphoton_delta_r = np.nan
        zeppenfeld_aa = np.nan
    selected_leptons = list(leptons[:4])
    four_lepton = combined_kinematics(selected_leptons) if len(selected_leptons) == 4 else combined_kinematics([])
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    candidates = []
    if len(selected_leptons) == 4:
        for first, second in pairings:
            pairs = ([selected_leptons[first[0]], selected_leptons[first[1]]], [selected_leptons[second[0]], selected_leptons[second[1]]])
            if all(a["pdg_id"] == -b["pdg_id"] for a, b in pairs):
                systems = [combined_kinematics(pair) for pair in pairs]
                candidates.append((sum(abs(system["mass"] - 91.1876) for system in systems), systems))
    systems = min(candidates, key=lambda item: item[0])[1] if candidates else [
        combined_kinematics(selected_leptons[:2]), combined_kinematics(selected_leptons[2:4])
    ]
    systems.sort(key=lambda system: abs(system["mass"] - 91.1876))
    z1, z2 = systems
    visible = [p for p in final if abs(p["pdg_id"]) not in INVISIBLE_PDGS]
    obs = {
        "e_j1": j1["e"], "pt_j1": j1["pt"], "phi_j1": j1["phi"], "eta_j1": j1["eta"],
        "e_j2": j2["e"], "pt_j2": j2["pt"], "phi_j2": j2["phi"], "eta_j2": j2["eta"],
        "delta_eta_jj": delta_eta_jj, "abs_delta_eta_jj": abs(delta_eta_jj),
        "delta_phi_jj": signed_delta_phi, "abs_delta_phi_jj": abs(signed_delta_phi), "m_jj": invariant_mass([j1, j2]),
        "e_a1": a1["e"] if a1 is not None else np.nan, "pt_a1": a1["pt"] if a1 is not None else np.nan,
        "phi_a1": a1["phi"] if a1 is not None else np.nan, "eta_a1": a1["eta"] if a1 is not None else np.nan,
        "e_a2": a2["e"] if a2 is not None else np.nan, "pt_a2": a2["pt"] if a2 is not None else np.nan,
        "phi_a2": a2["phi"] if a2 is not None else np.nan, "eta_a2": a2["eta"] if a2 is not None else np.nan,
        "delta_r_aa": diphoton_delta_r, "pt_aa": diphoton["pt"], "eta_aa": diphoton["eta"], "m_aa": diphoton["mass"],
        "met": math.hypot(-sum(p["px"] for p in visible), -sum(p["py"] for p in visible)),
        "visible_ht": sum(pt(p["px"], p["py"]) for p in visible), "zeppenfeld_aa": zeppenfeld_aa,
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


def build_cuts(cut_config: Mapping[str, Any]):
    cuts = []
    for feature, key in [
        ("pt_j1", "min_pt_j1"),
        ("pt_j2", "min_pt_j2"),
        ("pt_a1", "min_pt_a1"),
        ("pt_a2", "min_pt_a2"),
        ("m_jj", "min_m_jj"),
        ("abs_delta_eta_jj", "min_abs_delta_eta_jj"),
        ("delta_r_aa", "min_delta_r_aa"),
    ]:
        if cut_config.get(key) is not None:
            cuts.append(lambda obs, objects, feature=feature, value=float(cut_config[key]): np.isfinite(obs[feature]) and obs[feature] >= value)
    if cut_config.get("min_jets") is not None:
        cuts.append(lambda obs, objects, value=int(cut_config["min_jets"]): objects["n_jets"] >= value)
    if cut_config.get("min_photons") is not None:
        cuts.append(lambda obs, objects, value=int(cut_config["min_photons"]): objects["n_photons"] >= value)
    if cut_config.get("min_leptons") is not None:
        cuts.append(lambda obs, objects, value=int(cut_config["min_leptons"]): objects["n_leptons"] >= value)
    if cut_config.get("max_abs_eta_j") is not None:
        cuts.append(lambda obs, objects, value=float(cut_config["max_abs_eta_j"]): abs(obs["eta_j1"]) <= value and abs(obs["eta_j2"]) <= value)
    if cut_config.get("max_abs_eta_a") is not None:
        cuts.append(lambda obs, objects, value=float(cut_config["max_abs_eta_a"]): np.isfinite(obs["eta_a1"]) and np.isfinite(obs["eta_a2"]) and abs(obs["eta_a1"]) <= value and abs(obs["eta_a2"]) <= value)
    if cut_config.get("m_aa_window") is not None:
        low, high = map(float, cut_config["m_aa_window"])
        cuts.append(lambda obs, objects, low=low, high=high: np.isfinite(obs["m_aa"]) and low <= obs["m_aa"] <= high)
    if bool(cut_config.get("opposite_hemisphere_jets")):
        cuts.append(lambda obs, objects: obs["eta_j1"] * obs["eta_j2"] < 0.0)
    if cut_config.get("min_delta_r_jj") is not None:
        cuts.append(lambda obs, objects, value=float(cut_config["min_delta_r_jj"]): delta_r(objects["jets"][0], objects["jets"][1]) >= value)
    if cut_config.get("min_delta_r_ja") is not None:
        cuts.append(lambda obs, objects, value=float(cut_config["min_delta_r_ja"]): all(delta_r(jet, photon) >= value for jet in objects["jets"][:2] for photon in objects["photons"][:2]))
    return cuts


def load_preparation_settings(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    prep_path = resolve_config_path(config, config.get("preparation_config", DEFAULT_PREPARATION_CONFIG_PATH))
    with prep_path.open("r", encoding="utf-8-sig") as handle:
        prep = json.load(handle)
    feature_columns = config.get("feature_columns") or prep["physics"]["feature_columns"]
    prep_settings = prep["preparation"]
    smearing = {**prep_settings.get("smearing", {}), **config.get("smearing", {})}
    cut_config = {**prep_settings.get("cuts", {}), **config.get("cuts", {})}
    return list(feature_columns), smearing, cut_config


def build_features(lhe_path: Path, config: Mapping[str, Any], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    feature_columns, smearing, cut_config = load_preparation_settings(config)
    rules = active_smearing_rules(smearing)
    cuts = build_cuts(cut_config)
    rng = np.random.default_rng(int(smearing.get("seed", seed)))
    rows = []
    total_events = 0
    for event in parse_lhe_events(lhe_path):
        total_events += 1
        particles = [smear_particle(particle, rules, rng) for particle in event["particles"]]
        obs, objects = extract_observables(particles)
        if obs is None or not all(bool(cut(obs, objects)) for cut in cuts):
            continue
        rows.append([obs[column] for column in feature_columns])
    if not rows:
        raise ValueError(f"No events from {lhe_path} passed validation smearing and cuts.")
    metadata = {
        "feature_columns": feature_columns,
        "generated_events_read": total_events,
        "accepted_events": len(rows),
        "acceptance": len(rows) / total_events if total_events else 0.0,
    }
    return np.asarray(rows, dtype=np.float32), metadata


def run_madgraph_for_theta(
    config: Mapping[str, Any],
    theta: ThetaPoint,
    process_dir: Path,
    output_root: Path,
    log_dir: Path,
    index: int,
) -> dict[str, Any]:
    run_dir = output_root / theta.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("random_seed", 42)) + int(config.get("random_seed_stride", 1000)) * index
    nevents = int(config["N_generated"])
    run_name = f"validation_{index:03d}_{theta.tag}"

    configure_run_card(config, process_dir, seed, nevents)
    configure_param_card(config, process_dir, theta)

    generate_events = process_dir / "bin" / "generate_events"
    command = f'yes 0 | "{generate_events}" "{run_name}" -f'
    run_command(["bash", "-lc", command], log_dir / f"{run_name}.log", cwd=process_dir)

    event_file = find_event_file(process_dir / "Events" / run_name, str(config.get("lhe_filename", "unweighted_events.lhe.gz")))
    dataset_path = copy_event_file(event_file, run_dir)
    features, feature_metadata = build_features(dataset_path, config, seed)
    feature_path = run_dir / str(config.get("feature_filename", "features.npy"))
    np.save(feature_path, features)

    metadata = {
        "theta_true": [theta.c1, theta.c2],
        "theta_tag": theta.tag,
        "N_generated": nevents,
        "random_seed": seed,
        "process": config["process"],
        "event_file": portable_project_path(config, dataset_path),
        "feature_file": portable_project_path(config, feature_path),
        "madgraph_log": portable_project_path(config, log_dir / f"{run_name}.log"),
        **feature_metadata,
    }
    write_text(run_dir / "metadata.json", json.dumps(metadata, indent=2))
    print(
        f"{theta.tag}: accepted {metadata['accepted_events']:,}/{metadata['generated_events_read']:,} events",
        flush=True,
    )
    return metadata


def generate_validation_events(config_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Generate one validation dataset for each configured true point."""

    config = load_config(config_path)
    output_root = resolve_config_path(config, config.get("output_dir", "validation_events")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    process_dir, log_dir = make_process_dir(config, output_root)

    theta_points = [ThetaPoint.from_value(point) for point in config["theta_true"]]
    results = [
        run_madgraph_for_theta(config, point, process_dir, output_root, log_dir, index)
        for index, point in enumerate(theta_points)
    ]
    manifest_path = output_root / "manifest.json"
    write_text(manifest_path, json.dumps({"datasets": results}, indent=2))
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the validation event generation JSON config.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    results = generate_validation_events(args.config)
    print(json.dumps({"datasets": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
