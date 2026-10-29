from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
LATENT_DIR = PROJECT_ROOT / "experiments" / "step_03_latent_models"

for _path in (str(PROJECT_ROOT), str(SRC_DIR), str(LATENT_DIR), str(THIS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import step_02_latent_support_analysis as core
except ImportError as exc:
    raise ImportError(
        "Place this file in experiments/step_05_compare_metrics/ so it can "
        "import step_02_latent_support_analysis.py."
    ) from exc

from choice_generators import (
    M1Hyperbolic,
    M2Hyperboloid,
    M3DirectDifference,
    M4Exponential,
    M5DoubleExponential,
    M6ConstantSensitivity,
    M7AdditiveUtility,
    M8ProportionalDifference,
    M9PriorityLexicographicSemiorder,
)

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


TRIAL_DATA_NAMES = (
    "pub_Konstantinidis2020",
    "pub_Thrailkill2023",
)
REAL_DATA_NAMES = ("r1",)
ALL_SIMULATOR_MODELS = tuple(f"m{i}" for i in range(1, 10))


REPRESENTATION_TRAIN_TRIAL_DATA_NAME = "general_400"


PROTOCOLS = {
    "generalization_m4_m8": ("m4", "m5", "m6", "m7", "m8"),
    "recovery_m1_m3": ("m1", "m2", "m3"),
}


COARSE_FAMILIES = {
    "option_wise": ("m1", "m2", "m4", "m5", "m6", "m7"),
    "attribute_wise": ("m3", "m8"),
    "heuristic": ("m9",),
}

GENERALIZATION_TRAIN_MODELS = PROTOCOLS["generalization_m4_m8"]
GENERALIZATION_HELDOUT_MODELS = ("m1", "m2", "m3")
NOVEL_HEURISTIC_MODELS = ("m9",)


REPRESENTATION_KINDS = (
    "nn1",
    "nn1b",
    "classifier",
    "nn1c",
    "nn1bc",
    "parameter_decoder",
)


BETA = 0.1
CONSISTENCY_WEIGHT = 0.2
CLASSIFICATION_WEIGHT = 1.0
REPRESENTATION_HPARAMS: dict[str, dict] = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_BATCH_SIZE = 256
SEED = int(getattr(core, "SEED", 31))


MAX_SUBJECTS_PER_MODEL = 180
MAX_REAL_SUBJECTS = 250
MAX_FUNCTION_GRID_TRIALS = 500

PRIMARY_BEHAVIOR_DISTANCE = "functional_rmse"
BEHAVIOR_DISTANCE_METRICS = (
    "functional_rmse",
    "functional_mae",
)
LATENT_DISTANCE_METRIC = "euclidean"
LOCAL_NEIGHBOR_K = 10


RESAMPLE_REPEATS = 250
RESAMPLE_FRACTION = 0.80


MANTEL_PERMUTATIONS = 499
MANTEL_SCOPES = {
    "training_family_option",
    "training_all",
    "heldout_known",
    "heldout_plus_heuristic",
    "real_observed",
}


RUN_CLUSTERING = True
CLUSTER_SUBJECTS_PER_MODEL = 150
CLUSTER_MIXTURES = {
    "heldout_three_mechanisms": ("m1", "m3", "m9"),
    "heldout_four_models": ("m1", "m2", "m3", "m9"),
}
CLUSTER_RANDOM_REPEATS = 20


RUN_INTERPOLATION = True
INTERPOLATION_STEPS = 11
INTERPOLATION_PAIRS = {
    "within_option_m4": ("m4", "m4"),
    "training_cross_family_m4_m8": ("m4", "m8"),
    "heldout_known_m1_m3": ("m1", "m3"),
    "heldout_to_heuristic_m1_m9": ("m1", "m9"),
}


PLOT_PAIR_SAMPLE = 5000
SAVE_PLOTS = True
SHOW_FIGURES = False
OUTPUT_ROOT_NAME = "latent_geometry_validation"


CHOICE_MODEL_FACTORIES = {
    "m1": M1Hyperbolic,
    "m2": M2Hyperboloid,
    "m3": M3DirectDifference,
    "m4": M4Exponential,
    "m5": M5DoubleExponential,
    "m6": M6ConstantSensitivity,
    "m7": M7AdditiveUtility,
    "m8": M8ProportionalDifference,
    "m9": M9PriorityLexicographicSemiorder,
}

RECONSTRUCTION_KINDS = frozenset({"nn1", "nn1b", "nn1c", "nn1bc"})
NP_ONLY_KINDS = frozenset({"nn1", "nn1b"})

REPORT_LINES: list[str] = []


def log(message=""):
    text = str(message)
    print(text)
    REPORT_LINES.append(text)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_frame(frame: pd.DataFrame | None, path: Path):
    if frame is None:
        frame = pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def safe_json_dump(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def stable_rng(*parts) -> np.random.Generator:
    value = SEED
    for part in parts:
        text = str(part)
        for char in text:
            value = (value * 131 + ord(char)) % (2**32 - 1)
    return np.random.default_rng(value)


def coarse_family(model_name: str) -> str:
    for family_name, model_names in COARSE_FAMILIES.items():
        if model_name in model_names:
            return family_name
    return "unknown"


def configure_core_module():
    core.REPRESENTATION_TRAIN_TRIAL_DATA_NAME = (
        REPRESENTATION_TRAIN_TRIAL_DATA_NAME
    )
    core.BETA = BETA
    core.CONSISTENCY_WEIGHT = CONSISTENCY_WEIGHT
    core.CLASSIFICATION_WEIGHT = CLASSIFICATION_WEIGHT
    core.REPRESENTATION_HPARAMS = dict(REPRESENTATION_HPARAMS)
    core.DEVICE = DEVICE
    core.INFERENCE_BATCH_SIZE = INFERENCE_BATCH_SIZE


def model_test_indices(experiment, model_name: str, cap=None) -> np.ndarray:
    indices = np.asarray(
        core.indices_for_data_split(experiment, model_name, "test"),
        dtype=np.int64,
    )
    if cap is None:
        cap = MAX_SUBJECTS_PER_MODEL
    if cap is not None and int(cap) > 0 and len(indices) > int(cap):
        rng = stable_rng("select", model_name, len(indices), cap)
        indices = np.sort(rng.choice(indices, size=int(cap), replace=False))
    return indices


def real_indices(experiment, data_name: str) -> np.ndarray:
    indices = np.asarray(core.all_indices_for_data(experiment, data_name), dtype=np.int64)
    if MAX_REAL_SUBJECTS > 0 and len(indices) > MAX_REAL_SUBJECTS:
        rng = stable_rng("real", data_name)
        indices = np.sort(
            rng.choice(indices, size=MAX_REAL_SUBJECTS, replace=False)
        )
    return indices


def select_models_balanced(experiment, model_names, cap=None):
    model_names = tuple(model_names)
    pieces = []
    labels = []
    for model_name in model_names:
        idx = model_test_indices(experiment, model_name, cap=cap)
        pieces.append(idx)
        labels.extend([model_name] * len(idx))
    if not pieces:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=object)
    return np.concatenate(pieces), np.asarray(labels, dtype=object)


def common_trial_grid(experiment) -> np.ndarray:
    trials = experiment.choice_data.trials.detach().cpu().numpy()
    mask = experiment.choice_data.trial_mask.detach().cpu().numpy().astype(bool)

    if trials.ndim == 2:
        grid = np.asarray(trials, dtype=np.float64)
    elif trials.ndim == 3:


        pooled = trials[mask]
        rounded = np.round(np.asarray(pooled, dtype=np.float64), decimals=10)
        grid = np.unique(rounded, axis=0)
    else:
        raise ValueError("Unexpected trial layout.")

    if MAX_FUNCTION_GRID_TRIALS > 0 and len(grid) > MAX_FUNCTION_GRID_TRIALS:
        rng = stable_rng("grid", len(grid), MAX_FUNCTION_GRID_TRIALS)
        keep = np.sort(
            rng.choice(
                np.arange(len(grid)),
                size=MAX_FUNCTION_GRID_TRIALS,
                replace=False,
            )
        )
        grid = grid[keep]

    if grid.ndim != 2 or grid.shape[1] != 4:
        raise ValueError(
            "This analysis currently expects intertemporal-choice trials with "
            "four columns: amount_ss, delay_ss, amount_ll, delay_ll."
        )
    return grid


def model_probability_fingerprint(
    experiment,
    combined_index: int,
    model_name: str,
    grid: np.ndarray,
) -> np.ndarray:
    if model_name not in CHOICE_MODEL_FACTORIES:
        raise ValueError(f"No simulator implementation registered for {model_name}.")

    source_index = int(
        core.source_indices(
            experiment, np.asarray([combined_index], dtype=np.int64)
        )[0]
    )
    parameter_values = (
        experiment.parameters_by_model[model_name][source_index]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    parameter_names = list(experiment.parameter_names_by_model[model_name])
    parameters = dict(zip(parameter_names, parameter_values, strict=True))
    model = CHOICE_MODEL_FACTORIES[model_name]()
    probability = model._calculate_p_ll(grid, parameters)
    probability = np.asarray(probability, dtype=np.float64)
    return np.clip(probability, 0.0, 1.0)


def simulator_fingerprint_matrix(
    experiment,
    combined_indices: np.ndarray,
    model_labels: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    rows = []
    for combined_index, model_name in zip(
        combined_indices, model_labels, strict=True
    ):
        rows.append(
            model_probability_fingerprint(
                experiment,
                int(combined_index),
                str(model_name),
                grid,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def functional_distance_matrix(
    fingerprints: np.ndarray,
    metric: str,
) -> np.ndarray:
    fingerprints = np.asarray(fingerprints, dtype=np.float64)
    if len(fingerprints) < 2:
        return np.zeros((len(fingerprints), len(fingerprints)), dtype=np.float64)
    if metric == "functional_rmse":
        condensed = pdist(fingerprints, metric="euclidean") / math.sqrt(
            fingerprints.shape[1]
        )
    elif metric == "functional_mae":
        condensed = pdist(fingerprints, metric="cityblock") / float(
            fingerprints.shape[1]
        )
    else:
        raise ValueError(metric)
    return squareform(condensed)


def real_choice_distance_matrix(experiment, combined_indices: np.ndarray) -> np.ndarray:
    combined_indices = np.asarray(combined_indices, dtype=np.int64)
    choices = (
        experiment.choice_data.choices[combined_indices, :, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    mask = (
        experiment.choice_data.trial_mask[combined_indices]
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )
    n = len(combined_indices)
    output = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            valid = mask[i] & mask[j]
            if not valid.any():
                value = np.nan
            else:
                value = float(
                    np.sqrt(np.mean((choices[i, valid] - choices[j, valid]) ** 2))
                )
            output[i, j] = output[j, i] = value
    return output


@dataclass
class RepresentationOutput:
    label: str
    kind: str
    rep: object
    latent_raw: np.ndarray
    latent_scaled: np.ndarray


def load_protocol_representations(experiment, train_models):
    outputs: dict[str, RepresentationOutput] = {}
    for kind in REPRESENTATION_KINDS:
        try:
            rep = core.load_representation(kind, train_models)
        except Exception as exc:
            log(f"[skip] failed to load {kind} for {train_models}: {exc}")
            continue
        if rep is None:
            continue
        try:
            latent, _ = core.encode_representation(rep, experiment)
            scale = core.fit_latent_scale(latent, experiment, train_models)
            latent_scaled = scale.transform(latent)
        except Exception as exc:
            log(f"[skip] failed to encode {kind} for {train_models}: {exc}")
            continue
        outputs[rep.label] = RepresentationOutput(
            label=rep.label,
            kind=kind,
            rep=rep,
            latent_raw=np.asarray(latent, dtype=np.float64),
            latent_scaled=np.asarray(latent_scaled, dtype=np.float64),
        )
        log(
            f"  encoded {rep.label:<18} | train={','.join(train_models)} | "
            f"z={latent.shape}"
        )
    return outputs


def latent_distance_matrix(latent: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float64)
    if len(latent) < 2:
        return np.zeros((len(latent), len(latent)), dtype=np.float64)
    return squareform(pdist(latent, metric=LATENT_DISTANCE_METRIC))


def upper_triangle_values(matrix: np.ndarray):
    matrix = np.asarray(matrix, dtype=np.float64)
    tri = np.triu_indices(len(matrix), k=1)
    values = matrix[tri]
    return values, tri


def distance_spearman(a: np.ndarray, b: np.ndarray) -> float:
    av, tri = upper_triangle_values(a)
    bv = np.asarray(b, dtype=np.float64)[tri]
    valid = np.isfinite(av) & np.isfinite(bv)
    if valid.sum() < 3:
        return np.nan
    rho = spearmanr(av[valid], bv[valid]).statistic
    return float(rho) if np.isfinite(rho) else np.nan


def neighborhood_overlap(a: np.ndarray, b: np.ndarray, k=LOCAL_NEIGHBOR_K) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return np.nan
    k = min(max(1, int(k)), len(a) - 1)
    a_work = a.copy()
    b_work = b.copy()
    np.fill_diagonal(a_work, np.inf)
    np.fill_diagonal(b_work, np.inf)
    a_nn = np.argpartition(a_work, kth=k - 1, axis=1)[:, :k]
    b_nn = np.argpartition(b_work, kth=k - 1, axis=1)[:, :k]
    recalls = []
    for row_a, row_b in zip(a_nn, b_nn, strict=True):
        recalls.append(len(set(row_a.tolist()).intersection(row_b.tolist())) / k)
    return float(np.mean(recalls))


def subsample_statistics(
    latent_dist: np.ndarray,
    behavior_dist: np.ndarray,
    repeats=RESAMPLE_REPEATS,
    fraction=RESAMPLE_FRACTION,
    rng=None,
):
    if rng is None:
        rng = stable_rng("subsample")
    n = len(latent_dist)
    if n < 4 or repeats <= 0:
        return {
            "rho_low": np.nan,
            "rho_high": np.nan,
            "overlap_low": np.nan,
            "overlap_high": np.nan,
            "rho_samples": np.empty(0),
            "overlap_samples": np.empty(0),
        }
    size = min(n, max(4, int(round(n * fraction))))
    rho_samples = []
    overlap_samples = []
    for _ in range(int(repeats)):
        idx = np.sort(rng.choice(n, size=size, replace=False))
        ld = latent_dist[np.ix_(idx, idx)]
        bd = behavior_dist[np.ix_(idx, idx)]
        rho = distance_spearman(ld, bd)
        overlap = neighborhood_overlap(ld, bd)
        if np.isfinite(rho):
            rho_samples.append(rho)
        if np.isfinite(overlap):
            overlap_samples.append(overlap)
    rho_samples = np.asarray(rho_samples, dtype=np.float64)
    overlap_samples = np.asarray(overlap_samples, dtype=np.float64)

    def ci(values):
        if len(values) == 0:
            return (np.nan, np.nan)
        return tuple(np.quantile(values, [0.025, 0.975]).tolist())

    rho_low, rho_high = ci(rho_samples)
    overlap_low, overlap_high = ci(overlap_samples)
    return {
        "rho_low": rho_low,
        "rho_high": rho_high,
        "overlap_low": overlap_low,
        "overlap_high": overlap_high,
        "rho_samples": rho_samples,
        "overlap_samples": overlap_samples,
    }


def mantel_permutation_p(
    latent_dist: np.ndarray,
    behavior_dist: np.ndarray,
    permutations=MANTEL_PERMUTATIONS,
    rng=None,
) -> float:
    if permutations <= 0 or len(latent_dist) < 4:
        return np.nan
    if rng is None:
        rng = stable_rng("mantel")
    observed = distance_spearman(latent_dist, behavior_dist)
    if not np.isfinite(observed):
        return np.nan
    exceed = 0
    n = len(latent_dist)
    for _ in range(int(permutations)):
        perm = rng.permutation(n)
        permuted = behavior_dist[np.ix_(perm, perm)]
        value = distance_spearman(latent_dist, permuted)
        if np.isfinite(value) and abs(value) >= abs(observed):
            exceed += 1
    return float((exceed + 1) / (int(permutations) + 1))


def paired_representation_difference(
    latent_a: np.ndarray,
    latent_b: np.ndarray,
    behavior_dist: np.ndarray,
    repeats=RESAMPLE_REPEATS,
    fraction=RESAMPLE_FRACTION,
    rng=None,
):
    if rng is None:
        rng = stable_rng("paired_rep_diff")
    n = len(behavior_dist)
    estimate_rho = distance_spearman(latent_a, behavior_dist) - distance_spearman(
        latent_b, behavior_dist
    )
    estimate_overlap = neighborhood_overlap(
        latent_a, behavior_dist
    ) - neighborhood_overlap(latent_b, behavior_dist)

    if n < 4 or repeats <= 0:
        return {
            "delta_rho": estimate_rho,
            "delta_rho_low": np.nan,
            "delta_rho_high": np.nan,
            "p_delta_rho_gt_0": np.nan,
            "delta_overlap": estimate_overlap,
            "delta_overlap_low": np.nan,
            "delta_overlap_high": np.nan,
            "p_delta_overlap_gt_0": np.nan,
        }

    size = min(n, max(4, int(round(n * fraction))))
    deltas_rho = []
    deltas_overlap = []
    for _ in range(int(repeats)):
        idx = np.sort(rng.choice(n, size=size, replace=False))
        bd = behavior_dist[np.ix_(idx, idx)]
        la = latent_a[np.ix_(idx, idx)]
        lb = latent_b[np.ix_(idx, idx)]
        ra = distance_spearman(la, bd)
        rb = distance_spearman(lb, bd)
        oa = neighborhood_overlap(la, bd)
        ob = neighborhood_overlap(lb, bd)
        if np.isfinite(ra) and np.isfinite(rb):
            deltas_rho.append(ra - rb)
        if np.isfinite(oa) and np.isfinite(ob):
            deltas_overlap.append(oa - ob)

    deltas_rho = np.asarray(deltas_rho, dtype=np.float64)
    deltas_overlap = np.asarray(deltas_overlap, dtype=np.float64)

    def summarize(values):
        if len(values) == 0:
            return np.nan, np.nan, np.nan
        low, high = np.quantile(values, [0.025, 0.975])
        p_gt = float((np.sum(values > 0.0) + 0.5 * np.sum(values == 0.0)) / len(values))
        return float(low), float(high), p_gt

    rho_low, rho_high, rho_gt = summarize(deltas_rho)
    overlap_low, overlap_high, overlap_gt = summarize(deltas_overlap)
    return {
        "delta_rho": float(estimate_rho),
        "delta_rho_low": rho_low,
        "delta_rho_high": rho_high,
        "p_delta_rho_gt_0": rho_gt,
        "delta_overlap": float(estimate_overlap),
        "delta_overlap_low": overlap_low,
        "delta_overlap_high": overlap_high,
        "p_delta_overlap_gt_0": overlap_gt,
    }


def geometry_scopes_for_protocol(protocol_name: str, train_models):
    scopes = []


    evaluate_models = list(train_models)
    if protocol_name == "generalization_m4_m8":
        evaluate_models += list(GENERALIZATION_HELDOUT_MODELS)
        evaluate_models += list(NOVEL_HEURISTIC_MODELS)
    elif protocol_name == "recovery_m1_m3":
        evaluate_models += list(NOVEL_HEURISTIC_MODELS)
    evaluate_models = tuple(dict.fromkeys(evaluate_models))

    for model_name in evaluate_models:
        scopes.append(("within_model", f"model_{model_name}", (model_name,)))

    if protocol_name == "generalization_m4_m8":
        scopes.extend(
            [
                (
                    "training_family_option",
                    "training_option_m4_m7",
                    ("m4", "m5", "m6", "m7"),
                ),
                (
                    "training_all",
                    "training_m4_m8",
                    ("m4", "m5", "m6", "m7", "m8"),
                ),
                (
                    "heldout_known",
                    "heldout_m1_m3",
                    ("m1", "m2", "m3"),
                ),
                (
                    "heldout_plus_heuristic",
                    "heldout_m1_m3_plus_m9",
                    ("m1", "m2", "m3", "m9"),
                ),
            ]
        )
    else:
        scopes.extend(
            [
                (
                    "training_all",
                    "training_m1_m3",
                    ("m1", "m2", "m3"),
                ),
                (
                    "heldout_plus_heuristic",
                    "training_m1_m3_plus_m9",
                    ("m1", "m2", "m3", "m9"),
                ),
            ]
        )
    return scopes


def run_simulated_geometry(
    trial_data_name: str,
    protocol_name: str,
    train_models,
    experiment,
    grid: np.ndarray,
    rep_outputs: dict[str, RepresentationOutput],
    plot_pdf: PdfPages | None,
):
    summary_rows = []
    comparison_rows = []
    scope_cache = {}

    for scope, group_name, model_names in geometry_scopes_for_protocol(
        protocol_name, train_models
    ):
        combined_indices, labels = select_models_balanced(
            experiment, model_names
        )
        if len(combined_indices) < 4:
            log(f"  [skip] {group_name}: only {len(combined_indices)} subjects")
            continue

        fingerprints = simulator_fingerprint_matrix(
            experiment, combined_indices, labels, grid
        )
        behavior_matrices = {
            metric: functional_distance_matrix(fingerprints, metric)
            for metric in BEHAVIOR_DISTANCE_METRICS
        }
        latent_matrices = {
            rep_label: latent_distance_matrix(
                output.latent_scaled[combined_indices]
            )
            for rep_label, output in rep_outputs.items()
        }
        scope_cache[group_name] = {
            "indices": combined_indices,
            "labels": labels,
            "fingerprints": fingerprints,
            "behavior": behavior_matrices,
            "latent": latent_matrices,
        }

        log(
            f"  geometry {group_name:<32} | n={len(combined_indices):4d} | "
            f"models={','.join(model_names)}"
        )

        for behavior_metric, behavior_dist in behavior_matrices.items():
            for rep_label, latent_dist in latent_matrices.items():
                rho = distance_spearman(latent_dist, behavior_dist)
                overlap = neighborhood_overlap(latent_dist, behavior_dist)
                interval = subsample_statistics(
                    latent_dist,
                    behavior_dist,
                    rng=stable_rng(
                        trial_data_name,
                        protocol_name,
                        group_name,
                        behavior_metric,
                        rep_label,
                    ),
                )
                mantel_p = (
                    mantel_permutation_p(
                        latent_dist,
                        behavior_dist,
                        rng=stable_rng(
                            "mantel",
                            trial_data_name,
                            protocol_name,
                            group_name,
                            behavior_metric,
                            rep_label,
                        ),
                    )
                    if scope in MANTEL_SCOPES
                    and behavior_metric == PRIMARY_BEHAVIOR_DISTANCE
                    else np.nan
                )

                summary_rows.append(
                    {
                        "trial_data_name": trial_data_name,
                        "protocol": protocol_name,
                        "encoder_train_models": ",".join(train_models),
                        "scope": scope,
                        "group": group_name,
                        "models": ",".join(model_names),
                        "n_models": len(model_names),
                        "n_subjects": len(combined_indices),
                        "n_pairs": len(combined_indices)
                        * (len(combined_indices) - 1)
                        // 2,
                        "representation": rep_label,
                        "behavior_distance": behavior_metric,
                        "latent_distance": LATENT_DISTANCE_METRIC,
                        "spearman_rho": rho,
                        "rho_subsample_ci_low": interval["rho_low"],
                        "rho_subsample_ci_high": interval["rho_high"],
                        "mantel_p_two_sided": mantel_p,
                        "neighbor_k": min(
                            LOCAL_NEIGHBOR_K, len(combined_indices) - 1
                        ),
                        "neighbor_overlap": overlap,
                        "neighbor_overlap_ci_low": interval["overlap_low"],
                        "neighbor_overlap_ci_high": interval["overlap_high"],
                    }
                )

                if (
                    plot_pdf is not None
                    and behavior_metric == PRIMARY_BEHAVIOR_DISTANCE
                ):
                    make_pair_scatter_plot(
                        latent_dist,
                        behavior_dist,
                        title=(
                            f"{trial_data_name} | {protocol_name} | "
                            f"{group_name} | {rep_label}"
                        ),
                        rho=rho,
                        pdf=plot_pdf,
                        rng=stable_rng(
                            "plot",
                            trial_data_name,
                            protocol_name,
                            group_name,
                            rep_label,
                        ),
                    )


            if "classifier" in latent_matrices:
                classifier_dist = latent_matrices["classifier"]
                for rep_label in ("nn1", "nn1b", "nn1c", "nn1bc"):
                    if rep_label not in latent_matrices:
                        continue
                    result = paired_representation_difference(
                        latent_matrices[rep_label],
                        classifier_dist,
                        behavior_dist,
                        rng=stable_rng(
                            "delta",
                            trial_data_name,
                            protocol_name,
                            group_name,
                            behavior_metric,
                            rep_label,
                        ),
                    )
                    comparison_rows.append(
                        {
                            "trial_data_name": trial_data_name,
                            "protocol": protocol_name,
                            "scope": scope,
                            "group": group_name,
                            "models": ",".join(model_names),
                            "n_subjects": len(combined_indices),
                            "behavior_distance": behavior_metric,
                            "representation_a": rep_label,
                            "representation_b": "classifier",
                            **result,
                        }
                    )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(comparison_rows),
        scope_cache,
    )


def run_real_geometry(
    trial_data_name,
    protocol_name,
    train_models,
    experiment,
    rep_outputs,
    plot_pdf=None,
):
    rows = []
    comparisons = []
    for real_name in REAL_DATA_NAMES:
        if real_name not in experiment.subject_indices_by_data:
            continue
        indices = real_indices(experiment, real_name)
        if len(indices) < 4:
            continue
        behavior_dist = real_choice_distance_matrix(experiment, indices)
        latent_matrices = {
            rep_label: latent_distance_matrix(output.latent_scaled[indices])
            for rep_label, output in rep_outputs.items()
        }

        for rep_label, latent_dist in latent_matrices.items():
            rho = distance_spearman(latent_dist, behavior_dist)
            overlap = neighborhood_overlap(latent_dist, behavior_dist)
            interval = subsample_statistics(
                latent_dist,
                behavior_dist,
                rng=stable_rng(
                    "real",
                    trial_data_name,
                    protocol_name,
                    real_name,
                    rep_label,
                ),
            )
            mantel_p = mantel_permutation_p(
                latent_dist,
                behavior_dist,
                rng=stable_rng(
                    "real_mantel",
                    trial_data_name,
                    protocol_name,
                    real_name,
                    rep_label,
                ),
            )
            rows.append(
                {
                    "trial_data_name": trial_data_name,
                    "protocol": protocol_name,
                    "encoder_train_models": ",".join(train_models),
                    "scope": "real_observed",
                    "group": real_name,
                    "n_subjects": len(indices),
                    "n_pairs": len(indices) * (len(indices) - 1) // 2,
                    "representation": rep_label,
                    "behavior_distance": "observed_choice_rmse",
                    "latent_distance": LATENT_DISTANCE_METRIC,
                    "spearman_rho": rho,
                    "rho_subsample_ci_low": interval["rho_low"],
                    "rho_subsample_ci_high": interval["rho_high"],
                    "mantel_p_two_sided": mantel_p,
                    "neighbor_overlap": overlap,
                    "neighbor_overlap_ci_low": interval["overlap_low"],
                    "neighbor_overlap_ci_high": interval["overlap_high"],
                }
            )
            if plot_pdf is not None:
                make_pair_scatter_plot(
                    latent_dist,
                    behavior_dist,
                    title=(
                        f"REAL | {trial_data_name} | {protocol_name} | "
                        f"{real_name} | {rep_label}"
                    ),
                    rho=rho,
                    pdf=plot_pdf,
                    rng=stable_rng(
                        "real_plot",
                        trial_data_name,
                        protocol_name,
                        real_name,
                        rep_label,
                    ),
                )

        if "classifier" in latent_matrices:
            classifier_dist = latent_matrices["classifier"]
            for rep_label in ("nn1", "nn1b", "nn1c", "nn1bc"):
                if rep_label not in latent_matrices:
                    continue
                result = paired_representation_difference(
                    latent_matrices[rep_label],
                    classifier_dist,
                    behavior_dist,
                    rng=stable_rng(
                        "real_delta",
                        trial_data_name,
                        protocol_name,
                        real_name,
                        rep_label,
                    ),
                )
                comparisons.append(
                    {
                        "trial_data_name": trial_data_name,
                        "protocol": protocol_name,
                        "scope": "real_observed",
                        "group": real_name,
                        "n_subjects": len(indices),
                        "behavior_distance": "observed_choice_rmse",
                        "representation_a": rep_label,
                        "representation_b": "classifier",
                        **result,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def run_clustering_analysis(
    trial_data_name,
    protocol_name,
    experiment,
    grid,
    rep_outputs,
):
    if not RUN_CLUSTERING or not SKLEARN_AVAILABLE:
        if RUN_CLUSTERING and not SKLEARN_AVAILABLE:
            log("  [skip] sklearn unavailable; clustering analysis omitted")
        return pd.DataFrame()
    if protocol_name != "generalization_m4_m8":
        return pd.DataFrame()

    rows = []
    for mixture_name, model_names in CLUSTER_MIXTURES.items():
        indices, labels = select_models_balanced(
            experiment,
            model_names,
            cap=CLUSTER_SUBJECTS_PER_MODEL,
        )
        if len(indices) < len(model_names) * 3:
            continue
        unique_models = list(model_names)
        target = np.asarray(
            [unique_models.index(str(label)) for label in labels],
            dtype=np.int64,
        )
        n_clusters = len(unique_models)


        fingerprints = simulator_fingerprint_matrix(
            experiment, indices, labels, grid
        )
        feature_sets = {
            "true_behavior_fingerprint": fingerprints,
            **{
                rep_label: output.latent_scaled[indices]
                for rep_label, output in rep_outputs.items()
            },
        }

        for rep_label, features in feature_sets.items():
            features = np.asarray(features, dtype=np.float64)

            center = features.mean(axis=0, keepdims=True)
            scale = features.std(axis=0, ddof=0, keepdims=True)
            scale = np.where(scale > 1e-8, scale, 1.0)
            x = (features - center) / scale

            for method in ("kmeans", "gmm"):
                ari_values = []
                nmi_values = []
                silhouette_values = []
                for repeat in range(CLUSTER_RANDOM_REPEATS):
                    random_state = int(
                        stable_rng(
                            "cluster",
                            trial_data_name,
                            mixture_name,
                            rep_label,
                            method,
                            repeat,
                        ).integers(0, 2**31 - 1)
                    )
                    if method == "kmeans":
                        fitted = KMeans(
                            n_clusters=n_clusters,
                            n_init=20,
                            random_state=random_state,
                        ).fit_predict(x)
                    else:
                        fitted = GaussianMixture(
                            n_components=n_clusters,
                            covariance_type="full",
                            n_init=5,
                            random_state=random_state,
                            reg_covar=1e-5,
                        ).fit_predict(x)
                    ari_values.append(adjusted_rand_score(target, fitted))
                    nmi_values.append(normalized_mutual_info_score(target, fitted))
                    try:
                        silhouette_values.append(silhouette_score(x, fitted))
                    except Exception:
                        silhouette_values.append(np.nan)

                rows.append(
                    {
                        "trial_data_name": trial_data_name,
                        "protocol": protocol_name,
                        "mixture": mixture_name,
                        "models": ",".join(model_names),
                        "n_subjects": len(indices),
                        "n_clusters": n_clusters,
                        "representation": rep_label,
                        "clustering_method": method,
                        "ari_mean": float(np.nanmean(ari_values)),
                        "ari_sd": float(np.nanstd(ari_values)),
                        "nmi_mean": float(np.nanmean(nmi_values)),
                        "nmi_sd": float(np.nanstd(nmi_values)),
                        "silhouette_mean": float(np.nanmean(silhouette_values)),
                    }
                )

    return pd.DataFrame(rows)


def standardized_grid_for_rep(rep, grid: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(grid, dtype=torch.float32)
    standardized = core.apply_checkpoint_standardizer(
        tensor,
        rep.checkpoint.get("standardizer"),
    )
    return standardized.to(DEVICE)


@torch.no_grad()
def decode_probability_grid(rep, latent_rows: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if rep.kind not in core.NP_LATENT_KINDS | core.NP_CLASSIFIER_KINDS:
        raise ValueError("Representation does not have a choice decoder.")
    latent = torch.as_tensor(latent_rows, dtype=torch.float32, device=DEVICE)
    target_grid = standardized_grid_for_rep(rep, grid)
    target_trials = target_grid.unsqueeze(0).expand(len(latent), -1, -1)
    target_mask = torch.ones(
        (len(latent), len(grid)), dtype=torch.bool, device=DEVICE
    )
    logits = rep.model.decode(target_trials, latent, target_mask)
    probabilities = torch.sigmoid(logits[..., 0]).cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64)


def choose_interpolation_subject(experiment, model_name, which=0) -> int:
    indices = model_test_indices(experiment, model_name, cap=None)
    if len(indices) == 0:
        raise ValueError(f"No test subjects for {model_name}.")
    rng = stable_rng("interp_subject", model_name, which)
    return int(rng.choice(indices))


def run_interpolation_analysis(
    trial_data_name,
    protocol_name,
    experiment,
    grid,
    rep_outputs,
    output_dir: Path,
):
    if not RUN_INTERPOLATION:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    path_rows = []
    interp_dir = ensure_dir(output_dir / "interpolation")
    t_values = np.linspace(0.0, 1.0, INTERPOLATION_STEPS)

    for rep_label, output in rep_outputs.items():
        if output.kind not in RECONSTRUCTION_KINDS:
            continue
        rep = output.rep
        for pair_name, (model_a, model_b) in INTERPOLATION_PAIRS.items():
            try:
                idx_a = choose_interpolation_subject(experiment, model_a, which=0)
                idx_b = choose_interpolation_subject(
                    experiment,
                    model_b,
                    which=1 if model_a == model_b else 0,
                )
            except Exception as exc:
                log(f"  [skip] interpolation {pair_name}: {exc}")
                continue
            if idx_a == idx_b:
                candidates = model_test_indices(experiment, model_b, cap=None)
                alternatives = candidates[candidates != idx_a]
                if len(alternatives):
                    idx_b = int(alternatives[0])

            z_a = output.latent_raw[idx_a]
            z_b = output.latent_raw[idx_b]
            z_path = np.asarray(
                [(1.0 - t) * z_a + t * z_b for t in t_values],
                dtype=np.float64,
            )
            decoded = decode_probability_grid(rep, z_path, grid)

            true_a = model_probability_fingerprint(
                experiment, idx_a, model_a, grid
            )
            true_b = model_probability_fingerprint(
                experiment, idx_b, model_b, grid
            )
            endpoint_a_rmse = float(
                np.sqrt(np.mean((decoded[0] - true_a) ** 2))
            )
            endpoint_b_rmse = float(
                np.sqrt(np.mean((decoded[-1] - true_b) ** 2))
            )

            step_rmse = np.sqrt(
                np.mean(np.diff(decoded, axis=0) ** 2, axis=1)
            )
            chord = np.asarray(
                [
                    (1.0 - t) * decoded[0] + t * decoded[-1]
                    for t in t_values
                ]
            )
            chord_deviation = np.sqrt(
                np.mean((decoded - chord) ** 2, axis=1)
            )

            summary_rows.append(
                {
                    "trial_data_name": trial_data_name,
                    "protocol": protocol_name,
                    "representation": rep_label,
                    "pair": pair_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "subject_index_a": idx_a,
                    "subject_index_b": idx_b,
                    "endpoint_true_rmse_a": endpoint_a_rmse,
                    "endpoint_true_rmse_b": endpoint_b_rmse,
                    "mean_step_rmse": float(np.mean(step_rmse)),
                    "max_step_rmse": float(np.max(step_rmse)),
                    "mean_chord_deviation": float(np.mean(chord_deviation)),
                    "max_chord_deviation": float(np.max(chord_deviation)),
                    "decoded_endpoint_distance": float(
                        np.sqrt(np.mean((decoded[-1] - decoded[0]) ** 2))
                    ),
                    "true_endpoint_distance": float(
                        np.sqrt(np.mean((true_b - true_a) ** 2))
                    ),
                }
            )

            for step_index, t in enumerate(t_values):
                path_rows.append(
                    {
                        "trial_data_name": trial_data_name,
                        "protocol": protocol_name,
                        "representation": rep_label,
                        "pair": pair_name,
                        "step": step_index,
                        "t": float(t),
                        "mean_p_ll": float(decoded[step_index].mean()),
                        "distance_to_decoded_a": float(
                            np.sqrt(
                                np.mean(
                                    (decoded[step_index] - decoded[0]) ** 2
                                )
                            )
                        ),
                        "distance_to_decoded_b": float(
                            np.sqrt(
                                np.mean(
                                    (decoded[step_index] - decoded[-1]) ** 2
                                )
                            )
                        ),
                        "chord_deviation": float(chord_deviation[step_index]),
                    }
                )

            if SAVE_PLOTS:
                fig, ax = plt.subplots(figsize=(7.0, 4.0))
                image = ax.imshow(
                    decoded,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                )
                ax.set_xlabel("Common trial-grid index")
                ax.set_ylabel("Latent interpolation step")
                ax.set_yticks(np.arange(len(t_values)))
                ax.set_yticklabels([f"{t:.1f}" for t in t_values])
                ax.set_title(
                    f"{rep_label}: {model_a} $\\rightarrow$ {model_b}"
                )
                fig.colorbar(image, ax=ax, label="Decoded P(LL)")
                fig.tight_layout()
                fig.savefig(
                    interp_dir / f"{rep_label}__{pair_name}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                if SHOW_FIGURES:
                    plt.show()
                plt.close(fig)

    return pd.DataFrame(summary_rows), pd.DataFrame(path_rows)


def make_pair_scatter_plot(
    latent_dist,
    behavior_dist,
    title,
    rho,
    pdf: PdfPages,
    rng,
):
    latent_values, tri = upper_triangle_values(latent_dist)
    behavior_values = behavior_dist[tri]
    valid = np.isfinite(latent_values) & np.isfinite(behavior_values)
    x = latent_values[valid]
    y = behavior_values[valid]
    if len(x) == 0:
        return
    if PLOT_PAIR_SAMPLE > 0 and len(x) > PLOT_PAIR_SAMPLE:
        keep = rng.choice(len(x), size=PLOT_PAIR_SAMPLE, replace=False)
        x = x[keep]
        y = y[keep]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(x, y, s=8, alpha=0.25)
    ax.set_xlabel("Latent distance")
    ax.set_ylabel("Behavioral-function distance")
    ax.set_title(f"{title}\nSpearman $\\rho$={rho:.3f}")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def run_trial_design(trial_data_name: str):
    log("=" * 88)
    log(f"TRIAL DESIGN: {trial_data_name}")
    log("=" * 88)

    data_names = tuple(REAL_DATA_NAMES) + ALL_SIMULATOR_MODELS
    experiment = core.load_experiment_data(
        trial_data_name,
        data_names=data_names,
        seed=SEED,
        full_real=True,
    )
    grid = common_trial_grid(experiment)
    log(f"Common function grid: {grid.shape[0]} trials")

    trial_dir = ensure_dir(
        PROJECT_ROOT
        / "output"
        / OUTPUT_ROOT_NAME
        / trial_data_name
        / f"s{SEED}"
    )
    np.save(trial_dir / "common_function_grid.npy", grid)

    all_geometry = []
    all_comparisons = []
    all_real = []
    all_real_comparisons = []
    all_clustering = []
    all_interpolation = []
    all_interpolation_path = []
    checkpoint_rows = []

    for protocol_name, train_models in PROTOCOLS.items():
        log("")
        log(f"PROTOCOL: {protocol_name} | train={','.join(train_models)}")
        protocol_dir = ensure_dir(trial_dir / protocol_name)

        rep_outputs = load_protocol_representations(experiment, train_models)
        for label, output in rep_outputs.items():
            checkpoint_rows.append(
                {
                    "trial_data_name": trial_data_name,
                    "protocol": protocol_name,
                    "train_models": ",".join(train_models),
                    "representation": label,
                    "kind": output.kind,
                    "checkpoint_path": str(output.rep.checkpoint_path),
                }
            )
        if not rep_outputs:
            log("  [skip] no representations available")
            continue

        pdf_path = protocol_dir / "pairwise_geometry_scatter.pdf"
        pdf_context = PdfPages(pdf_path) if SAVE_PLOTS else None
        try:
            geometry, comparisons, _ = run_simulated_geometry(
                trial_data_name,
                protocol_name,
                train_models,
                experiment,
                grid,
                rep_outputs,
                pdf_context,
            )
            real_geometry, real_comparisons = run_real_geometry(
                trial_data_name,
                protocol_name,
                train_models,
                experiment,
                rep_outputs,
                pdf_context,
            )
        finally:
            if pdf_context is not None:
                pdf_context.close()

        clustering = run_clustering_analysis(
            trial_data_name,
            protocol_name,
            experiment,
            grid,
            rep_outputs,
        )
        interpolation, interpolation_path = run_interpolation_analysis(
            trial_data_name,
            protocol_name,
            experiment,
            grid,
            rep_outputs,
            protocol_dir,
        )

        save_frame(geometry, protocol_dir / "geometry_fidelity_summary.csv")
        save_frame(
            comparisons,
            protocol_dir / "np_vs_classifier_geometry_comparisons.csv",
        )
        save_frame(real_geometry, protocol_dir / "real_observed_geometry.csv")
        save_frame(
            real_comparisons,
            protocol_dir / "real_np_vs_classifier_comparisons.csv",
        )
        save_frame(clustering, protocol_dir / "synthetic_heterogeneity_clustering.csv")
        save_frame(interpolation, protocol_dir / "np_generative_interpolation_summary.csv")
        save_frame(
            interpolation_path,
            protocol_dir / "np_generative_interpolation_path.csv",
        )

        all_geometry.append(geometry)
        all_comparisons.append(comparisons)
        all_real.append(real_geometry)
        all_real_comparisons.append(real_comparisons)
        all_clustering.append(clustering)
        all_interpolation.append(interpolation)
        all_interpolation_path.append(interpolation_path)

        print_protocol_highlights(
            trial_data_name,
            protocol_name,
            geometry,
            comparisons,
            real_geometry,
            clustering,
        )

    def concat(parts):
        parts = [x for x in parts if x is not None and not x.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    aggregate = {
        "geometry_fidelity_summary.csv": concat(all_geometry),
        "np_vs_classifier_geometry_comparisons.csv": concat(all_comparisons),
        "real_observed_geometry.csv": concat(all_real),
        "real_np_vs_classifier_comparisons.csv": concat(all_real_comparisons),
        "synthetic_heterogeneity_clustering.csv": concat(all_clustering),
        "np_generative_interpolation_summary.csv": concat(all_interpolation),
        "np_generative_interpolation_path.csv": concat(all_interpolation_path),
        "checkpoint_coverage.csv": pd.DataFrame(checkpoint_rows),
    }
    for filename, frame in aggregate.items():
        save_frame(frame, trial_dir / filename)

    config = {
        "trial_data_name": trial_data_name,
        "seed": SEED,
        "representation_train_trial_data_name": REPRESENTATION_TRAIN_TRIAL_DATA_NAME,
        "protocols": {k: list(v) for k, v in PROTOCOLS.items()},
        "representations": list(REPRESENTATION_KINDS),
        "coarse_families": {k: list(v) for k, v in COARSE_FAMILIES.items()},
        "max_subjects_per_model": MAX_SUBJECTS_PER_MODEL,
        "function_grid_trials": len(grid),
        "behavior_distance_metrics": list(BEHAVIOR_DISTANCE_METRICS),
        "latent_distance_metric": LATENT_DISTANCE_METRIC,
        "local_neighbor_k": LOCAL_NEIGHBOR_K,
        "resample_repeats": RESAMPLE_REPEATS,
        "resample_fraction": RESAMPLE_FRACTION,
        "mantel_permutations": MANTEL_PERMUTATIONS,
        "cluster_mixtures": {k: list(v) for k, v in CLUSTER_MIXTURES.items()},
        "interpolation_pairs": {k: list(v) for k, v in INTERPOLATION_PAIRS.items()},
    }
    safe_json_dump(config, trial_dir / "config.json")
    return aggregate


def print_protocol_highlights(
    trial_data_name,
    protocol_name,
    geometry,
    comparisons,
    real_geometry,
    clustering,
):
    log("")
    log("  Highlights")
    if geometry is not None and not geometry.empty:
        main = geometry[
            geometry["behavior_distance"] == PRIMARY_BEHAVIOR_DISTANCE
        ].copy()
        focus = main[
            main["scope"].isin(
                [
                    "training_family_option",
                    "heldout_known",
                    "heldout_plus_heuristic",
                ]
            )
        ]
        for group_name, group in focus.groupby("group", sort=False):
            ordered = group.sort_values("spearman_rho", ascending=False)
            best = ordered.iloc[0]
            log(
                f"    {group_name}: best rho={best['spearman_rho']:.3f} "
                f"({best['representation']}); "
                f"neighbor overlap={best['neighbor_overlap']:.3f}"
            )

    if comparisons is not None and not comparisons.empty:
        focus = comparisons[
            (comparisons["behavior_distance"] == PRIMARY_BEHAVIOR_DISTANCE)
            & comparisons["scope"].isin(
                ["within_model", "heldout_known", "heldout_plus_heuristic"]
            )
            & comparisons["representation_a"].isin(["nn1", "nn1b"])
        ].copy()
        if not focus.empty:
            positive = int((focus["delta_rho"] > 0).sum())
            log(
                f"    pure NP vs classifier: delta-rho positive in "
                f"{positive}/{len(focus)} evaluated scopes"
            )

    if real_geometry is not None and not real_geometry.empty:
        real = real_geometry.sort_values("spearman_rho", ascending=False)
        best = real.iloc[0]
        log(
            f"    real observed geometry: best rho={best['spearman_rho']:.3f} "
            f"({best['representation']})"
        )

    if clustering is not None and not clustering.empty:
        subset = clustering[
            clustering["clustering_method"] == "kmeans"
        ].sort_values("ari_mean", ascending=False)
        if not subset.empty:
            best = subset.iloc[0]
            log(
                f"    clustering: best ARI={best['ari_mean']:.3f} "
                f"({best['mixture']}, {best['representation']})"
            )


def main():
    configure_core_module()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    log("Latent geometry validation")
    log(f"Device: {DEVICE}")
    log(f"Trial designs: {TRIAL_DATA_NAMES}")
    log(f"Representations: {REPRESENTATION_KINDS}")
    log(
        "Primary scientific test: correspondence between standardized latent "
        "distance and noise-free simulator behavioral-function distance."
    )

    trial_outputs = {}
    for trial_data_name in TRIAL_DATA_NAMES:
        try:
            trial_outputs[trial_data_name] = run_trial_design(trial_data_name)
        except Exception as exc:
            log(f"[ERROR] {trial_data_name}: {type(exc).__name__}: {exc}")
            raise

    cross_dir = ensure_dir(
        PROJECT_ROOT / "output" / OUTPUT_ROOT_NAME / "aggregate"
    )
    aggregate_keys = (
        "geometry_fidelity_summary.csv",
        "np_vs_classifier_geometry_comparisons.csv",
        "real_observed_geometry.csv",
        "real_np_vs_classifier_comparisons.csv",
        "synthetic_heterogeneity_clustering.csv",
        "np_generative_interpolation_summary.csv",
        "checkpoint_coverage.csv",
    )
    for key in aggregate_keys:
        frames = [
            output[key]
            for output in trial_outputs.values()
            if key in output and output[key] is not None and not output[key].empty
        ]
        combined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )
        save_frame(combined, cross_dir / key)

    report_path = cross_dir / "results.txt"
    report_path.write_text("\n".join(REPORT_LINES) + "\n", encoding="utf-8")
    log("")
    log(f"Done. Aggregate outputs: {cross_dir}")


if __name__ == "__main__":
    main()
