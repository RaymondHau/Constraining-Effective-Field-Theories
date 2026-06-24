#!/usr/bin/env python3
"""Compare direct validation events with morphed prepared-event distributions.

The script reads direct validation ``features.npy`` files from
``validation_events.py`` and compares them with the proposal-corrected morphed
distribution implied by ``EFT_event_generation.py`` + ``EFT_prepare_samples.py``.
It writes one overlay plot per validation point and observable, plus CSV/JSON
summaries containing KL divergences and useful morphing diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_project_path(value: str | Path, base_dir: Path = PROJECT_DIR) -> Path:
    """Resolve paths that may be relative, native absolute, or WSL absolute."""

    path = Path(value).expanduser()
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = base_dir / path
        if candidate.exists():
            return candidate
        return candidate

    parts = path.parts
    if base_dir.name in parts:
        suffix = Path(*parts[parts.index(base_dir.name) + 1 :])
        candidate = base_dir / suffix
        if candidate.exists():
            return candidate
    return path


def config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    return dict(config.get(name, {}))


def theta_vector(theta: Mapping[str, float], operators: Sequence[str]) -> np.ndarray:
    return np.array([float(theta[name]) for name in operators], dtype=np.float64)


def theta_matrix(thetas: Sequence[Mapping[str, float]], operators: Sequence[str]) -> np.ndarray:
    return np.vstack([theta_vector(theta, operators) for theta in thetas])


def quadratic_basis(theta: np.ndarray, morphing_scale: Mapping[str, float], operators: Sequence[str]) -> np.ndarray:
    scale = theta_vector(morphing_scale, operators)
    scaled_theta = np.asarray(theta, dtype=np.float64) / scale
    c1 = scaled_theta[..., 0]
    c2 = scaled_theta[..., 1]
    return np.stack([np.ones_like(c1), c1, c2, c1 * c1, c1 * c2, c2 * c2], axis=-1)


def generated_events_by_benchmark(
    benchmark_names: Sequence[str],
    reference_benchmark: str,
    prep_config: Mapping[str, Any],
) -> dict[str, int]:
    non_reference = [name for name in benchmark_names if name != reference_benchmark]
    total = int(prep_config.get("paper_parameterized_generated_events", 10_000_000))
    events_per_basis = total // (2 * len(non_reference))
    return {
        reference_benchmark: events_per_basis * len(non_reference),
        **{name: events_per_basis for name in non_reference},
    }


def load_feature_frame(dataset: Mapping[str, Any], feature_columns: Sequence[str]) -> pd.DataFrame:
    feature_path = resolve_project_path(dataset["feature_file"])
    features = np.load(feature_path)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features in {feature_path}, got shape {features.shape}")
    if features.shape[1] != len(feature_columns):
        raise ValueError(
            f"{feature_path} has {features.shape[1]} columns, but manifest lists {len(feature_columns)}"
        )
    return pd.DataFrame(features, columns=feature_columns)


def normalized_probabilities(hist: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(hist, dtype=np.float64)
    total = values.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(values, dtype=np.float64) / max(len(values), 1)
    probs = values / total
    probs = np.maximum(probs, epsilon)
    return probs / probs.sum()


def kl_divergence(p_hist: np.ndarray, q_hist: np.ndarray, epsilon: float) -> float:
    p = normalized_probabilities(p_hist, epsilon)
    q = normalized_probabilities(q_hist, epsilon)
    return float(np.sum(p * np.log(p / q)))


def density_for_plot(hist: np.ndarray, bins: np.ndarray) -> np.ndarray:
    widths = np.diff(bins)
    area = np.sum(hist)
    if area <= 0.0:
        return np.zeros_like(hist, dtype=np.float64)
    return hist / area / widths


def robust_standardize_pair(p_values: np.ndarray, q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([p_values, q_values])
    center = np.nanmedian(combined, axis=0)
    q25, q75 = np.nanpercentile(combined, [25.0, 75.0], axis=0)
    scale = q75 - q25
    fallback = np.nanstd(combined, axis=0)
    scale = np.where((scale > 0.0) & np.isfinite(scale), scale, fallback)
    scale = np.where((scale > 0.0) & np.isfinite(scale), scale, 1.0)
    return (p_values - center) / scale, (q_values - center) / scale


def kth_neighbor_distances(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    chunk_size: int,
    exclude_self: bool = False,
) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be at least 1.")
    if reference.shape[0] <= k and exclude_self:
        raise ValueError("Need more reference rows than k for self-neighbor distances.")
    if reference.shape[0] < k and not exclude_self:
        raise ValueError("Need at least k reference rows.")

    kth_distances = np.empty(query.shape[0], dtype=np.float64)
    ref_norm = np.sum(reference * reference, axis=1)
    kth_index = k - 1
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        block = query[start:stop]
        dist2 = np.sum(block * block, axis=1)[:, None] + ref_norm[None, :] - 2.0 * block @ reference.T
        dist2 = np.maximum(dist2, 0.0)
        if exclude_self:
            row_indices = np.arange(start, stop)
            dist2[np.arange(stop - start), row_indices] = np.inf
        kth = np.partition(dist2, kth_index, axis=1)[:, kth_index]
        kth_distances[start:stop] = np.sqrt(np.maximum(kth, 0.0))
    return kth_distances


def knn_kl_divergence(
    p_values: np.ndarray,
    q_values: np.ndarray,
    k: int,
    chunk_size: int,
    epsilon: float,
) -> float:
    if p_values.shape[1] != q_values.shape[1]:
        raise ValueError("p_values and q_values must have the same feature dimension.")
    n = p_values.shape[0]
    m = q_values.shape[0]
    dimension = p_values.shape[1]
    if n <= k or m < k:
        raise ValueError(f"Need n > k and m >= k for kNN KL; got n={n}, m={m}, k={k}.")

    rho = kth_neighbor_distances(p_values, p_values, k=k, chunk_size=chunk_size, exclude_self=True)
    nu = kth_neighbor_distances(p_values, q_values, k=k, chunk_size=chunk_size, exclude_self=False)
    rho = np.maximum(rho, epsilon)
    nu = np.maximum(nu, epsilon)
    return float((dimension / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1.0)))


def collective_kl_summary(
    direct_df: pd.DataFrame,
    event_df: pd.DataFrame,
    probabilities: np.ndarray,
    feature_columns: Sequence[str],
    rng: np.random.Generator,
    k: int,
    max_samples: int | None,
    chunk_size: int,
    epsilon: float,
) -> dict[str, Any]:
    direct_matrix = direct_df[list(feature_columns)].to_numpy(dtype=np.float64)
    prepared_matrix = event_df[list(feature_columns)].to_numpy(dtype=np.float64)
    direct_mask = np.all(np.isfinite(direct_matrix), axis=1)
    prepared_mask = np.all(np.isfinite(prepared_matrix), axis=1) & np.isfinite(probabilities)
    direct_matrix = direct_matrix[direct_mask]
    prepared_matrix = prepared_matrix[prepared_mask]
    prepared_probabilities = np.asarray(probabilities[prepared_mask], dtype=np.float64)
    prepared_probabilities = np.where(prepared_probabilities > 0.0, prepared_probabilities, 0.0)
    prepared_probabilities /= prepared_probabilities.sum()

    baseline: dict[str, float | int] = {
        "validation_self_split_kl_a_to_b": np.nan,
        "validation_self_split_kl_b_to_a": np.nan,
        "validation_self_split_sample_size": 0,
    }
    split_size = len(direct_matrix) // 2
    if split_size > k:
        split_indices = rng.permutation(len(direct_matrix))
        split_a = direct_matrix[split_indices[:split_size]]
        split_b = direct_matrix[split_indices[split_size : split_size * 2]]
        split_a, split_b = robust_standardize_pair(split_a, split_b)
        baseline = {
            "validation_self_split_kl_a_to_b": knn_kl_divergence(
                split_a,
                split_b,
                k=k,
                chunk_size=chunk_size,
                epsilon=epsilon,
            ),
            "validation_self_split_kl_b_to_a": knn_kl_divergence(
                split_b,
                split_a,
                k=k,
                chunk_size=chunk_size,
                epsilon=epsilon,
            ),
            "validation_self_split_sample_size": int(split_size),
        }

    sample_size = len(direct_matrix)
    if max_samples is not None:
        sample_size = min(sample_size, int(max_samples))
    if sample_size <= k:
        raise ValueError(f"Need more than k={k} finite validation rows for collective KL.")

    if sample_size < len(direct_matrix):
        direct_indices = rng.choice(len(direct_matrix), size=sample_size, replace=False)
        direct_sample = direct_matrix[direct_indices]
    else:
        direct_sample = direct_matrix
    prepared_indices = rng.choice(
        len(prepared_matrix),
        size=sample_size,
        replace=True,
        p=prepared_probabilities,
    )
    prepared_sample = prepared_matrix[prepared_indices]

    direct_scaled, prepared_scaled = robust_standardize_pair(direct_sample, prepared_sample)
    kl = knn_kl_divergence(
        direct_scaled,
        prepared_scaled,
        k=k,
        chunk_size=chunk_size,
        epsilon=epsilon,
    )
    reverse_kl = knn_kl_divergence(
        prepared_scaled,
        direct_scaled,
        k=k,
        chunk_size=chunk_size,
        epsilon=epsilon,
    )
    return {
        "collective_kl_validation_to_morphed": kl,
        "collective_kl_morphed_to_validation": reverse_kl,
        "collective_kl_k": int(k),
        "collective_kl_observables": int(len(feature_columns)),
        "collective_kl_sample_size": int(sample_size),
        "finite_validation_rows": int(len(direct_matrix)),
        "finite_prepared_rows": int(len(prepared_matrix)),
        **baseline,
    }


def finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def build_bins(
    direct_values: np.ndarray,
    morphed_values: np.ndarray,
    bins: int,
    percentile_range: tuple[float, float],
) -> np.ndarray | None:
    combined = np.concatenate([direct_values, morphed_values])
    combined = combined[np.isfinite(combined)]
    if len(combined) == 0:
        return None
    low, high = np.percentile(combined, percentile_range)
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        low = float(np.min(combined))
        high = float(np.max(combined))
    if np.isclose(low, high):
        return None
    return np.linspace(low, high, bins + 1)


def setup_morphing(
    event_df: pd.DataFrame,
    config: Mapping[str, Any],
    generated_event_overrides: Mapping[str, int] | None,
) -> dict[str, Any]:
    physics = config_section(config, "physics")
    preparation = config_section(config, "preparation")
    operators = list(physics["eft_operators"])
    benchmark_points = {
        name: {op: float(value) for op, value in theta.items()}
        for name, theta in physics["benchmark_points"].items()
    }
    benchmark_names = list(benchmark_points)
    reference_benchmark = str(physics.get("reference_benchmark", benchmark_names[0]))
    weight_columns = [f"w_{name}" for name in benchmark_names]
    missing = [column for column in weight_columns if column not in event_df.columns]
    if missing:
        raise KeyError(f"Prepared event table is missing weight columns: {missing}")

    morphing_scale = dict(physics.get("morphing_theta_scale", {op: 1.0 for op in operators}))
    basis_at_benchmarks = quadratic_basis(
        theta_matrix([benchmark_points[name] for name in benchmark_names], operators),
        morphing_scale,
        operators,
    )
    morphing_matrix = np.linalg.pinv(basis_at_benchmarks)
    event_coefficients = event_df[weight_columns].to_numpy(dtype=np.float64) @ morphing_matrix.T

    generated_counts = generated_events_by_benchmark(benchmark_names, reference_benchmark, preparation)
    if generated_event_overrides:
        generated_counts.update({str(k): int(v) for k, v in generated_event_overrides.items()})

    target_epsilon = float(preparation.get("target_epsilon", 1.0e-30))
    sigma_benchmarks = np.array(
        [
            event_df.loc[event_df["source_benchmark"] == name, f"w_{name}"].sum()
            / max(float(generated_counts[name]), target_epsilon)
            for name in benchmark_names
        ],
        dtype=np.float64,
    )
    sigma_coefficients = morphing_matrix @ sigma_benchmarks

    counts = event_df["source_benchmark"].value_counts().reindex(benchmark_names, fill_value=0).to_numpy(dtype=np.float64)
    if counts.sum() <= 0.0:
        raise ValueError("Prepared event table has no recognized source_benchmark rows.")
    mixture_fractions = counts / counts.sum()
    proposal_density_proxy = np.zeros(len(event_df), dtype=np.float64)
    for fraction, name, sigma in zip(mixture_fractions, benchmark_names, sigma_benchmarks):
        proposal_density_proxy += (
            fraction
            * event_df[f"w_{name}"].to_numpy(dtype=np.float64)
            / max(float(sigma), target_epsilon)
        )
    proposal_density_proxy = np.maximum(proposal_density_proxy, target_epsilon)

    return {
        "operators": operators,
        "benchmark_names": benchmark_names,
        "generated_counts": generated_counts,
        "morphing_scale": morphing_scale,
        "condition_number": float(np.linalg.cond(basis_at_benchmarks)),
        "event_coefficients": event_coefficients,
        "sigma_coefficients": sigma_coefficients,
        "sigma_benchmarks": dict(zip(benchmark_names, sigma_benchmarks.tolist())),
        "proposal_density_proxy": proposal_density_proxy,
        "target_epsilon": target_epsilon,
    }


def morphed_probabilities(theta: Mapping[str, float], morphing: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    operators = morphing["operators"]
    theta_arr = theta_vector(theta, operators)
    theta_values = np.repeat(theta_arr[None, :], len(morphing["event_coefficients"]), axis=0)
    phi = quadratic_basis(theta_values, morphing["morphing_scale"], operators)
    weights_theta = np.einsum("ij,ij->i", morphing["event_coefficients"], phi)
    sigma_theta = float(
        quadratic_basis(theta_arr[None, :], morphing["morphing_scale"], operators)[0]
        @ morphing["sigma_coefficients"]
    )
    target_density = weights_theta / max(sigma_theta, morphing["target_epsilon"])
    probabilities = target_density / morphing["proposal_density_proxy"]

    raw_negative = int(np.sum(probabilities <= 0.0))
    raw_nonfinite = int(np.sum(~np.isfinite(probabilities)))
    probabilities = np.where(np.isfinite(probabilities) & (probabilities > 0.0), probabilities, 0.0)
    total = float(probabilities.sum())
    if total <= 0.0:
        probabilities = np.ones(len(probabilities), dtype=np.float64) / len(probabilities)
    else:
        probabilities = probabilities / total

    diagnostics = {
        "sigma_theta": sigma_theta,
        "raw_negative_or_zero_probability_rows": raw_negative,
        "raw_nonfinite_probability_rows": raw_nonfinite,
        "positive_probability_sum_before_norm": total,
    }
    return probabilities, diagnostics


def theta_from_dataset(
    dataset: Mapping[str, Any],
    operators: Sequence[str],
    validation_config: Mapping[str, Any] | None,
) -> dict[str, float]:
    theta_values = list(map(float, dataset["theta_true"]))
    names = list((validation_config or {}).get("coefficient_names", operators))
    if len(theta_values) != len(names):
        raise ValueError(f"Cannot map theta_true={theta_values} onto coefficient names={names}")
    named = dict(zip(names, theta_values))
    return {operator: float(named[operator]) for operator in operators}


def compare_dataset(
    dataset: Mapping[str, Any],
    event_df: pd.DataFrame,
    morphing: Mapping[str, Any],
    feature_columns: Sequence[str],
    validation_config: Mapping[str, Any] | None,
    output_dir: Path,
    bins: int,
    percentile_range: tuple[float, float],
    epsilon: float,
    rng: np.random.Generator,
    plot_overlays: bool,
    compute_collective_kl: bool,
    collective_kl_k: int,
    collective_kl_max_samples: int | None,
    collective_kl_chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    theta = theta_from_dataset(dataset, morphing["operators"], validation_config)
    probabilities, diagnostics = morphed_probabilities(theta, morphing)
    direct_df = load_feature_frame(dataset, feature_columns)

    tag = str(dataset.get("theta_tag", "_".join(f"{k}_{v:g}" for k, v in theta.items())))
    point_dir = output_dir / tag
    point_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for feature in feature_columns:
        direct_values = finite_values(direct_df, feature)
        morphed_values = finite_values(event_df, feature)
        finite_mask = np.isfinite(event_df[feature].to_numpy(dtype=np.float64))
        morphed_weights = probabilities[finite_mask]
        plot_bins = build_bins(direct_values, morphed_values, bins, percentile_range)
        if plot_bins is None:
            rows.append(
                {
                    "theta_tag": tag,
                    "feature": feature,
                    "kl_validation_to_morphed": np.nan,
                    "status": "skipped_constant_or_empty",
                }
            )
            continue

        direct_hist, _ = np.histogram(direct_values, bins=plot_bins)
        morphed_hist, _ = np.histogram(morphed_values, bins=plot_bins, weights=morphed_weights)
        kl_vm = kl_divergence(direct_hist, morphed_hist, epsilon)
        kl_mv = kl_divergence(morphed_hist, direct_hist, epsilon)
        if plot_overlays:
            plt = get_pyplot()
            centers = 0.5 * (plot_bins[:-1] + plot_bins[1:])
            plt.figure(figsize=(7.0, 4.2))
            plt.step(centers, density_for_plot(direct_hist, plot_bins), where="mid", label="validation direct")
            plt.step(centers, density_for_plot(morphed_hist, plot_bins), where="mid", label="morphed prepared")
            plt.xlabel(feature)
            plt.ylabel("normalized density")
            plt.title(f"{tag}: {feature}\nKL(validation || morphed) = {kl_vm:.4g}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(point_dir / f"{feature}.png", dpi=160)
            plt.close()

        rows.append(
            {
                "theta_tag": tag,
                "theta": json.dumps(theta, sort_keys=True),
                "feature": feature,
                "kl_validation_to_morphed": kl_vm,
                "kl_morphed_to_validation": kl_mv,
                "validation_entries": int(len(direct_values)),
                "prepared_entries": int(len(morphed_values)),
                "bin_low": float(plot_bins[0]),
                "bin_high": float(plot_bins[-1]),
                "bins": int(len(plot_bins) - 1),
                "status": "ok",
            }
        )

    point_summary = {
        "theta_tag": tag,
        "theta": theta,
        "validation_feature_file": str(resolve_project_path(dataset["feature_file"])),
        "validation_accepted_events": int(dataset.get("accepted_events", len(direct_df))),
        **diagnostics,
    }
    collective_summary = None
    if compute_collective_kl:
        collective_summary = {
            "theta_tag": tag,
            "theta": json.dumps(theta, sort_keys=True),
            **collective_kl_summary(
                direct_df=direct_df,
                event_df=event_df,
                probabilities=probabilities,
                feature_columns=feature_columns,
                rng=rng,
                k=collective_kl_k,
                max_samples=collective_kl_max_samples,
                chunk_size=collective_kl_chunk_size,
                epsilon=epsilon,
            ),
        }
        point_summary.update(collective_summary)
    return rows, point_summary, collective_summary


def load_generated_event_overrides(output_dir: Path) -> dict[str, int] | None:
    budget_path = output_dir / "generated_event_budget.csv"
    if not budget_path.exists():
        return None
    frame = pd.read_csv(budget_path)
    if not {"benchmark", "generated_events_requested"}.issubset(frame.columns):
        return None
    return dict(zip(frame["benchmark"].astype(str), frame["generated_events_requested"].astype(int)))


def parse_args() -> argparse.Namespace:
    process_config_dir = f"configs/{os.environ.get('EFT_PROCESS', 'WBF')}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", default="madgraph_work/validation_events/manifest.json")
    parser.add_argument("--validation-config", default=f"{process_config_dir}/validation_events_config.json")
    parser.add_argument("--sample-config", default=f"{process_config_dir}/sample_preparation.json")
    parser.add_argument("--event-table", default="table_outputs/prepared_events/end_to_end_events.csv")
    parser.add_argument("--sample-output-dir", default="table_outputs/madminer_style_training")
    parser.add_argument("--output-dir", default="table_outputs/validation_morphing_comparison")
    parser.add_argument("--plot-output-dir", default="plotting_outputs/validation_morphing")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--percentile-low", type=float, default=0.5)
    parser.add_argument("--percentile-high", type=float, default=99.5)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    parser.add_argument("--max-prepared-events", type=int, default=None, help="Optional debug subsample of prepared events.")
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--skip-plots", action="store_true", help="Compute KL tables without writing observable overlay PNGs.")
    parser.add_argument("--skip-collective-kl", action="store_true", help="Disable the full-observable kNN KL estimate.")
    parser.add_argument("--collective-kl-k", type=int, default=5, help="k for the k-nearest-neighbor collective KL estimator.")
    parser.add_argument(
        "--collective-kl-max-samples",
        type=int,
        default=None,
        help="Optional cap on validation/morphed rows used in each collective KL estimate.",
    )
    parser.add_argument(
        "--collective-kl-chunk-size",
        type=int,
        default=512,
        help="Distance-matrix chunk size for the collective KL estimator.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_json(resolve_project_path(args.validation_manifest))
    validation_config_path = resolve_project_path(args.validation_config)
    validation_config = load_json(validation_config_path) if validation_config_path.exists() else None
    sample_config = load_json(resolve_project_path(args.sample_config))
    output_dir = resolve_project_path(args.output_dir)
    plot_output_dir = resolve_project_path(args.plot_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(manifest["datasets"])
    if not datasets:
        raise ValueError("Validation manifest has no datasets.")
    feature_columns = list(datasets[0].get("feature_columns") or config_section(sample_config, "physics")["feature_columns"])

    event_table_path = resolve_project_path(args.event_table)
    use_columns = ["source_benchmark", *feature_columns]
    use_columns += [f"w_{name}" for name in config_section(sample_config, "physics")["benchmark_points"]]
    use_column_set = set(use_columns)
    event_df = pd.read_csv(event_table_path, usecols=lambda column: column in use_column_set)
    if args.max_prepared_events is not None and len(event_df) > args.max_prepared_events:
        event_df = event_df.sample(n=args.max_prepared_events, random_state=args.random_seed).reset_index(drop=True)

    generated_overrides = load_generated_event_overrides(resolve_project_path(args.sample_output_dir))
    morphing = setup_morphing(event_df, sample_config, generated_overrides)

    all_rows: list[dict[str, Any]] = []
    point_summaries: list[dict[str, Any]] = []
    collective_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.random_seed)
    percentile_range = (float(args.percentile_low), float(args.percentile_high))
    if not (0.0 <= percentile_range[0] < percentile_range[1] <= 100.0):
        raise ValueError("Require 0 <= percentile-low < percentile-high <= 100.")

    for dataset in datasets:
        rows, summary, collective_summary = compare_dataset(
            dataset=dataset,
            event_df=event_df,
            morphing=morphing,
            feature_columns=feature_columns,
            validation_config=validation_config,
            output_dir=plot_output_dir,
            bins=args.bins,
            percentile_range=percentile_range,
            epsilon=float(args.epsilon),
            rng=rng,
            plot_overlays=not args.skip_plots,
            compute_collective_kl=not args.skip_collective_kl,
            collective_kl_k=args.collective_kl_k,
            collective_kl_max_samples=args.collective_kl_max_samples,
            collective_kl_chunk_size=args.collective_kl_chunk_size,
        )
        all_rows.extend(rows)
        point_summaries.append(summary)
        if collective_summary is not None:
            collective_rows.append(collective_summary)

    summary_df = pd.DataFrame(all_rows)
    summary_path = output_dir / "kl_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    collective_path = output_dir / "collective_kl_summary.csv"
    collective_df = pd.DataFrame(collective_rows)
    if collective_rows:
        collective_df.to_csv(collective_path, index=False)

    metadata = {
        "validation_manifest": str(resolve_project_path(args.validation_manifest)),
        "sample_config": str(resolve_project_path(args.sample_config)),
        "event_table": str(event_table_path),
        "output_dir": str(output_dir),
        "plot_output_dir": str(plot_output_dir),
        "feature_columns": feature_columns,
        "morphing": {
            "operators": morphing["operators"],
            "benchmark_names": morphing["benchmark_names"],
            "generated_counts": morphing["generated_counts"],
            "condition_number": morphing["condition_number"],
            "sigma_benchmarks": morphing["sigma_benchmarks"],
        },
        "points": point_summaries,
        "mean_kl_validation_to_morphed": float(summary_df["kl_validation_to_morphed"].mean(skipna=True)),
        "median_kl_validation_to_morphed": float(summary_df["kl_validation_to_morphed"].median(skipna=True)),
    }
    if collective_rows:
        metadata["mean_collective_kl_validation_to_morphed"] = float(
            collective_df["collective_kl_validation_to_morphed"].mean(skipna=True)
        )
        metadata["median_collective_kl_validation_to_morphed"] = float(
            collective_df["collective_kl_validation_to_morphed"].median(skipna=True)
        )
    (output_dir / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote KL summary: {summary_path}")
    if collective_rows:
        print(f"Wrote collective KL summary: {collective_path}")
    if args.skip_plots:
        print("Skipped overlay plots.")
    else:
        print(f"Wrote plots under: {plot_output_dir}")
    print(
        "Mean KL(validation || morphed): "
        f"{metadata['mean_kl_validation_to_morphed']:.6g}; "
        "median: "
        f"{metadata['median_kl_validation_to_morphed']:.6g}"
    )
    if collective_rows:
        print(
            "Mean collective KL(validation || morphed): "
            f"{metadata['mean_collective_kl_validation_to_morphed']:.6g}; "
            "median: "
            f"{metadata['median_collective_kl_validation_to_morphed']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
