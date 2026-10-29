"""Validate latent-support model comparison and misspecification diagnostics.

This script is intentionally parallel to ``step_01_compare_metrics*.py`` but
answers a different set of scientific questions.  It uses a small, predeclared
set of latent-support scores and *calibrates each score within each candidate
model* before models are compared.

The four analyses are:

A. Held-out-model structural generalization
   Train representation on m4--m8, then embed held-out m1--m3.  Ask whether
   each held-out model is assigned to the correct *model family*.

B. Conventional model recovery
   Train representation on m1--m3 and compare latent-support recovery with BIC
   and the deterministic model-classifier output when those files/checkpoints
   are available.

C. Candidate-family misspecification / coverage
   Using the m4--m8 representation, define the discounted-value models
   m4--m6 as the candidate family.  Compare candidate-family compatibility for
   (i) held-out m4--m6 test subjects, (ii) unseen same-family m1--m2, and
   (iii) held-out different-family m3.  AUROC quantifies whether the support
   score detects the omitted family while accepting unseen models from the
   covered family.

D. Real-subject application
   With m1--m3 as candidate models, report the closest latent model and the
   *absolute* calibrated compatibility C_max for every real participant.  The
   simulations and real data are loaded from the same TRIAL_DATA_NAME, so the
   support comparison is design matched.

Primary latent-support scores
-----------------------------
``mean_euclidean``
    Naive baseline: mean Euclidean distance to all reference simulations.
``knn``
    Primary nonparametric support score: mean distance to the k nearest
    reference simulations.
``gaussian_nll``
    Simple probabilistic baseline: negative log density under a regularized
    full Gaussian fitted to the model's reference z cloud.
``param_manifold``
    Simulator-structured score: distance to a denoised parameterized surface
    theta -> E[z | theta, model].  The surface is fitted with a low-degree
    polynomial ridge map.  By default distance is evaluated to the fitted
    surface at the observed reference theta values (fast).  Optional continuous
    L-BFGS refinement can be enabled below.

Calibration
-----------
For each representation, candidate model, and score, the evaluation-model
subjects are split deterministically into:
    reference = train split,
    calibration = validation split,
    query = test split (or all real subjects).

A query raw score S_m(z) is converted to a model-specific compatibility:

    C_m(z) = [1 +

so larger C means "more typical of simulations from model m".  This empirical
calibration is what should be compared across models; raw distances are kept in
outputs for diagnostics but are not interpreted as probabilities.

Outputs are written under:
    output/latent_support_analysis/<trial-data>/s<seed>/

The script skips individual representation checkpoints or BIC fits that are
missing, rather than aborting the whole analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations_with_replacement
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
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LATENT_DIR = PROJECT_ROOT / "experiments" / "step_03_latent_models"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(LATENT_DIR))

from common import (
    SEED,
    choice_data_path,
    load_checkpoint,
    load_experiment_data,
    order_data_names,
    output_root,
)
from choice_generators.model_name_transfer import MODEL_METADATA
from model_classifier import (
    ChoiceModelClassifier,
    ModelClassifierConfig,
)
from neural_processes_choice import (
    ChoiceNPConfig,
    ChoiceNeuralProcess,
    load_choice_data,
)
from neural_processes_choice_classifier import (
    ChoiceNPClassifierConfig,
    ChoiceNeuralProcessClassifier,
)
from parameter_decoder import (
    ChoiceParameterDecoder,
    ParameterDecoderConfig,
)








TRIAL_DATA_NAME = "pub_Konstantinidis2020"
REAL_DATA_NAMES = ("r1",)
MODEL_DATA_NAMES = tuple(f"m{i}" for i in range(1, 9))


REPRESENTATION_TRAIN_TRIAL_DATA_NAME = "general_400"


GENERALIZATION_TRAIN_MODELS = ("m4", "m5", "m6", "m7", "m8")
GENERALIZATION_HELDOUT_MODELS = ("m1", "m2", "m3")


RECOVERY_TRAIN_MODELS = ("m1", "m2", "m3")
RECOVERY_QUERY_MODELS = ("m1", "m2", "m3")
RECOVERY_CANDIDATE_MODELS = ("m1", "m2", "m3")



MISSPEC_CANDIDATE_MODELS = ("m4", "m5", "m6")
MISSPEC_ID_MODELS = ("m4", "m5", "m6")
MISSPEC_SAME_FAMILY_UNSEEN = ("m1", "m2")
MISSPEC_OUT_FAMILY_UNSEEN = ("m3",)



REPRESENTATION_KINDS = (
    "np",
    "classifier",
    "np_classifier",
    "parameter_decoder",
)
BETA = 0.1
CONSISTENCY_WEIGHT = 0.2
CLASSIFICATION_WEIGHT = 1.0



REPRESENTATION_HPARAMS: dict[str, dict] = {}

CANONICAL_REPRESENTATION_KIND = {
    "np": "nn1b",
    "np_classifier": "nn1bc",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_BATCH_SIZE = 256
MAX_POINTS_PER_DATA = 1000


SCORE_KINDS = (
    "mean_euclidean",
    "knn",
    "gaussian_nll",
    "param_manifold",
)
PRIMARY_SCORE = "knn"
KNN_K = 10




STANDARDIZE_LATENT_AXES = True
LATENT_SCALE_FLOOR = 1e-6

GAUSSIAN_RIDGE = 1e-4
PARAM_MANIFOLD_POLY_DEGREE = 3
PARAM_MANIFOLD_RIDGE = 1e-4
PARAM_MANIFOLD_CONTINUOUS_REFINE = False
PARAM_MANIFOLD_MAXITER = 80


COMPATIBILITY_THRESHOLDS = (0.05, 0.01)
SHOW_FIGURES = True

OUTPUT_ROOT_NAME = "latent_support_analysis"






REPORT_LINES: list[str] = []


def log(message=""):
    text = str(message)
    print(text)
    REPORT_LINES.append(text)


def model_family(model_name: str) -> int:
    return int(MODEL_METADATA[model_name]["class"])


def model_new_name(model_name: str) -> str:
    return str(MODEL_METADATA[model_name]["new_name"])


def require_models(names) -> tuple[str, ...]:
    ordered = order_data_names(names)
    invalid = [name for name in ordered if name not in MODEL_METADATA]
    if invalid:
        raise ValueError(f"Unknown model names: {invalid}")
    return ordered


def limit_indices(indices: np.ndarray) -> np.ndarray:
    if MAX_POINTS_PER_DATA is None or int(MAX_POINTS_PER_DATA) <= 0:
        return indices
    return indices[: int(MAX_POINTS_PER_DATA)]


def indices_for_data_split(experiment, data_name: str, split: str) -> np.ndarray:
    data_indices = experiment.subject_indices_by_data[data_name]
    split_indices = experiment.subject_splits[split]
    selected = data_indices[torch.isin(data_indices, split_indices)]
    return limit_indices(selected.cpu().numpy().astype(np.int64))


def all_indices_for_data(experiment, data_name: str) -> np.ndarray:
    indices = experiment.subject_indices_by_data[data_name]
    return limit_indices(indices.cpu().numpy().astype(np.int64))


def source_indices(experiment, combined_indices: np.ndarray) -> np.ndarray:
    tensor_indices = torch.as_tensor(combined_indices, dtype=torch.long)
    return (
        experiment.metadata["source_subject_index"][tensor_indices]
        .cpu()
        .numpy()
        .astype(np.int64)
    )


def load_subject_ids(trial_data_name: str, data_name: str) -> np.ndarray:
    dataset = load_choice_data(choice_data_path(trial_data_name, data_name))
    if "subject_ids" in dataset.metadata:
        return np.asarray(dataset.metadata["subject_ids"]).astype(str)
    return np.asarray([str(i) for i in range(dataset.n_subjects)], dtype=str)


def batch_trials(data, indices: torch.Tensor) -> torch.Tensor:
    if data.trials.ndim == 2:
        return data.trials.unsqueeze(0).expand(len(indices), -1, -1)
    return data.trials[indices]


def apply_checkpoint_standardizer(trials, standardizer):
    if standardizer is None:
        return trials
    mean = standardizer["mean"]
    scale = standardizer["scale"]
    if trials.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"Trial dimension {trials.shape[-1]} does not match checkpoint "
            f"standardizer dimension {mean.shape[0]}."
        )
    return (trials - mean) / scale


def output_dir() -> Path:
    path = (
        PROJECT_ROOT
        / "output"
        / OUTPUT_ROOT_NAME
        / TRIAL_DATA_NAME
        / f"s{SEED}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path







@dataclass
class LoadedRepresentation:
    label: str
    kind: str
    train_models: tuple[str, ...]
    checkpoint_path: Path
    model: torch.nn.Module
    checkpoint: dict
    classifier_model_names: tuple[str, ...] | None = None



NP_LATENT_KINDS = frozenset({"nn1", "nn1b", "np"})
NP_CLASSIFIER_KINDS = frozenset({"nn1c", "nn1bc", "np_classifier"})


def representation_label(kind: str) -> str:
    return {
        "nn1": "nn1",
        "nn1b": "nn1b",
        "np": "nn1b",
        "classifier": "classifier",
        "nn1c": "nn1c",
        "nn1bc": "nn1bc",
        "np_classifier": "nn1bc",
        "parameter_decoder": "parameter_decoder",
    }[kind]


def representation_hparams_for(kind: str) -> dict:
    canonical = CANONICAL_REPRESENTATION_KIND.get(kind, kind)
    specific = REPRESENTATION_HPARAMS.get(canonical, {})
    return {
        "beta": specific.get("beta", BETA),
        "consistency_weight": specific.get(
            "consistency_weight", CONSISTENCY_WEIGHT
        ),
        "classification_weight": specific.get(
            "classification_weight", CLASSIFICATION_WEIGHT
        ),
    }


def representation_checkpoint_path(kind: str, train_models) -> Path:
    train_models = require_models(train_models)
    hparams = representation_hparams_for(kind)
    beta = hparams["beta"]
    consistency_weight = hparams["consistency_weight"]
    classification_weight = hparams["classification_weight"]
    if kind == "nn1":
        return (
            output_root(
                "neural_process",
                REPRESENTATION_TRAIN_TRIAL_DATA_NAME,
                train_models,
            )
            / f"b{beta:g}"
            / "checkpoint.pt"
        )
    if kind in {"nn1b", "np"}:
        return (
            output_root(
                "nn1b",
                REPRESENTATION_TRAIN_TRIAL_DATA_NAME,
                train_models,
            )
            / f"b{beta:g}_c{consistency_weight:g}"
            / "checkpoint.pt"
        )
    if kind == "classifier":
        return (
            output_root(
                "model_classifier",
                REPRESENTATION_TRAIN_TRIAL_DATA_NAME,
                train_models,
            )
            / "checkpoint.pt"
        )
    if kind == "nn1c":
        model_names = tuple(
            name for name in train_models if name.startswith("m")
        )
        if not model_names:
            model_names = train_models
        return (
            output_root(
                "nn1c",
                REPRESENTATION_TRAIN_TRIAL_DATA_NAME,
                model_names,
            )
            / f"b{beta:g}_cls{classification_weight:g}"
            / "checkpoint.pt"
        )
    if kind in {"nn1bc", "np_classifier"}:
        return (
            output_root(
                "nn1bc",
                REPRESENTATION_TRAIN_TRIAL_DATA_NAME,
                train_models,
            )
            / (
                f"b{beta:g}_c{consistency_weight:g}_"
                f"cls{classification_weight:g}"
            )
            / "checkpoint.pt"
        )
    if kind == "parameter_decoder":
        return (
            output_root(
                "parameter_decoder",
                REPRESENTATION_TRAIN_TRIAL_DATA_NAME,
                train_models,
            )
            / "checkpoint.pt"
        )
    raise ValueError(f"Unknown representation kind: {kind}")


def load_representation(kind: str, train_models) -> LoadedRepresentation | None:
    train_models = require_models(train_models)
    path = representation_checkpoint_path(kind, train_models)
    if not path.exists():
        log(f"[skip] missing {representation_label(kind)} checkpoint: {path}")
        return None
    checkpoint = load_checkpoint(path)
    config = checkpoint["model_config"]
    classifier_model_names = None
    if kind in NP_LATENT_KINDS:
        model = ChoiceNeuralProcess(ChoiceNPConfig(**config))
    elif kind == "classifier":
        model = ChoiceModelClassifier(ModelClassifierConfig(**config))
        extra = checkpoint.get("extra_state") or {}
        classifier_model_names = tuple(extra.get("model_names", train_models))
    elif kind in NP_CLASSIFIER_KINDS:
        model = ChoiceNeuralProcessClassifier(ChoiceNPClassifierConfig(**config))
        extra = checkpoint.get("extra_state") or {}
        classifier_model_names = tuple(extra.get("model_names", train_models))
    elif kind == "parameter_decoder":
        model = ChoiceParameterDecoder(ParameterDecoderConfig(**config))
    else:
        raise ValueError(kind)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    log(f"Loaded {representation_label(kind)}: {path}")
    return LoadedRepresentation(
        label=representation_label(kind),
        kind=kind,
        train_models=train_models,
        checkpoint_path=path,
        model=model,
        checkpoint=checkpoint,
        classifier_model_names=classifier_model_names,
    )


@torch.no_grad()
def encode_representation(rep: LoadedRepresentation, experiment):
    data = replace(
        experiment.choice_data,
        trials=apply_checkpoint_standardizer(
            experiment.choice_data.trials,
            rep.checkpoint.get("standardizer"),
        ),
    )
    all_indices = torch.arange(data.n_subjects, dtype=torch.long)
    latent_parts = []
    logits_parts = []
    for start in range(0, len(all_indices), INFERENCE_BATCH_SIZE):
        indices = all_indices[start : start + INFERENCE_BATCH_SIZE]
        trials = batch_trials(data, indices).to(DEVICE)
        choices = data.choices[indices].to(DEVICE)
        mask = data.trial_mask[indices].to(DEVICE)
        if rep.kind in NP_LATENT_KINDS | NP_CLASSIFIER_KINDS:
            mean, _ = rep.model.encode_context(trials, choices, mask)
            latent = mean
            if rep.kind in NP_CLASSIFIER_KINDS:
                logits_parts.append(rep.model.classify(mean).cpu())
        elif rep.kind == "classifier":
            latent = rep.model.encode(trials, choices, mask)
            logits_parts.append(rep.model.classifier_head(latent).cpu())
        elif rep.kind == "parameter_decoder":
            latent = rep.model.encode(trials, choices, mask)
        else:
            raise ValueError(rep.kind)
        latent_parts.append(latent.cpu())
    latent = torch.cat(latent_parts).numpy().astype(np.float64)
    probabilities = None
    if logits_parts:
        logits = torch.cat(logits_parts).numpy().astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    return latent, probabilities


@dataclass
class LatentScale:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, z):
        return (np.asarray(z, dtype=np.float64) - self.mean) / self.scale


def fit_latent_scale(latent: np.ndarray, experiment, train_models) -> LatentScale:
    ref_indices = np.concatenate(
        [indices_for_data_split(experiment, m, "train") for m in train_models]
    )
    values = latent[ref_indices]
    if not STANDARDIZE_LATENT_AXES:
        return LatentScale(
            mean=np.zeros(values.shape[1], dtype=np.float64),
            scale=np.ones(values.shape[1], dtype=np.float64),
        )
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > LATENT_SCALE_FLOOR, scale, 1.0)
    return LatentScale(mean=mean, scale=scale)







def euclidean_distance_matrix(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    squared_first = np.sum(first**2, axis=1, keepdims=True)
    squared_second = np.sum(second**2, axis=1)
    squared = squared_first + squared_second - 2.0 * (first @ second.T)
    return np.sqrt(np.maximum(squared, 0.0))


def pairwise_support_scores(query_z, reference_z):
    distances = euclidean_distance_matrix(query_z, reference_z)
    mean_distance = distances.mean(axis=1)
    k = min(max(1, int(KNN_K)), distances.shape[1])
    knn_distance = np.mean(
        np.partition(distances, k - 1, axis=1)[:, :k],
        axis=1,
    )
    return {
        "mean_euclidean": mean_distance,
        "knn": knn_distance,
    }


@dataclass
class GaussianSupport:
    mean: np.ndarray
    precision: np.ndarray
    logdet: float
    dimension: int


def fit_gaussian_support(reference_z) -> GaussianSupport:
    reference_z = np.asarray(reference_z, dtype=np.float64)
    center = reference_z.mean(axis=0)
    covariance = np.cov(reference_z, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]])
    covariance = covariance + GAUSSIAN_RIDGE * np.eye(covariance.shape[0])
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Regularized covariance is not positive definite.")
    return GaussianSupport(
        mean=center,
        precision=np.linalg.pinv(covariance),
        logdet=float(logdet),
        dimension=reference_z.shape[1],
    )


def gaussian_nll(query_z, support: GaussianSupport):
    delta = np.asarray(query_z, dtype=np.float64) - support.mean
    quadratic = np.einsum(
        "ij,jk,ik->i", delta, support.precision, delta
    )
    return 0.5 * (
        quadratic
        + support.logdet
        + support.dimension * np.log(2.0 * np.pi)
    )


def polynomial_powers(n_features, degree):
    powers = [tuple([0] * n_features)]
    for total_degree in range(1, int(degree) + 1):
        for combo in combinations_with_replacement(range(n_features), total_degree):
            exponent = [0] * n_features
            for feature_index in combo:
                exponent[feature_index] += 1
            powers.append(tuple(exponent))
    return np.asarray(powers, dtype=np.int64)


def polynomial_design(x, powers):
    x = np.asarray(x, dtype=np.float64)
    return np.prod(
        np.power(x[:, np.newaxis, :], powers[np.newaxis, :, :]),
        axis=2,
    )


@dataclass
class ParameterManifold:
    theta_mean: np.ndarray
    theta_scale: np.ndarray
    theta_std: np.ndarray
    powers: np.ndarray
    coefficients: np.ndarray
    bounds: np.ndarray
    fitted_reference_z: np.ndarray
    train_r2: float
    validation_r2: float

    def predict(self, theta_std):
        design = polynomial_design(np.atleast_2d(theta_std), self.powers)
        return design @ self.coefficients


def fit_parameter_manifold(theta_ref, z_ref, theta_val=None, z_val=None):
    theta_ref = np.asarray(theta_ref, dtype=np.float64)
    z_ref = np.asarray(z_ref, dtype=np.float64)
    if theta_ref.ndim != 2 or len(theta_ref) != len(z_ref) or len(theta_ref) < 2:
        return None
    theta_mean = theta_ref.mean(axis=0)
    theta_scale = theta_ref.std(axis=0, ddof=0)
    theta_scale = np.where(theta_scale > 1e-10, theta_scale, 1.0)
    theta_std = (theta_ref - theta_mean) / theta_scale
    powers = polynomial_powers(theta_ref.shape[1], PARAM_MANIFOLD_POLY_DEGREE)
    design = polynomial_design(theta_std, powers)
    penalty = PARAM_MANIFOLD_RIDGE * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ z_ref,
    )
    fitted = design @ coefficients

    def r2_score(actual, predicted):
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - actual.mean(axis=0)) ** 2))
        return np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

    train_r2 = r2_score(z_ref, fitted)
    validation_r2 = np.nan
    if theta_val is not None and z_val is not None and len(theta_val):
        theta_val_std = (np.asarray(theta_val) - theta_mean) / theta_scale
        val_pred = polynomial_design(theta_val_std, powers) @ coefficients
        validation_r2 = r2_score(np.asarray(z_val), val_pred)
    bounds = np.column_stack([theta_std.min(axis=0), theta_std.max(axis=0)])
    return ParameterManifold(
        theta_mean=theta_mean,
        theta_scale=theta_scale,
        theta_std=theta_std,
        powers=powers,
        coefficients=coefficients,
        bounds=bounds,
        fitted_reference_z=fitted,
        train_r2=float(train_r2),
        validation_r2=float(validation_r2),
    )


def parameter_manifold_distance(query_z, manifold: ParameterManifold | None):
    query_z = np.asarray(query_z, dtype=np.float64)
    if manifold is None:
        return np.full(len(query_z), np.nan, dtype=np.float64)

    start_distances = euclidean_distance_matrix(
        query_z, manifold.fitted_reference_z
    )
    nearest_rows = np.argmin(start_distances, axis=1)
    fast_distance = start_distances[
        np.arange(len(query_z)), nearest_rows
    ]
    if not PARAM_MANIFOLD_CONTINUOUS_REFINE:
        return fast_distance

    bounds = [tuple(item) for item in manifold.bounds]
    out = fast_distance.copy()
    for row, target_z in enumerate(query_z):
        start = manifold.theta_std[nearest_rows[row]].copy()

        def objective(theta_std):
            predicted = manifold.predict(theta_std)[0]
            delta = target_z - predicted
            return float(delta @ delta)

        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": PARAM_MANIFOLD_MAXITER},
        )
        if result.success and np.isfinite(result.fun):
            out[row] = math.sqrt(max(float(result.fun), 0.0))
    return out


def theta_for_combined_indices(experiment, model_name, combined_indices):
    src = source_indices(experiment, combined_indices)
    return (
        experiment.parameters_by_model[model_name][
            torch.as_tensor(src, dtype=torch.long)
        ]
        .cpu()
        .numpy()
        .astype(np.float64)
    )


@dataclass
class ModelSupport:
    model_name: str
    reference_z: np.ndarray
    gaussian: GaussianSupport
    parameter_manifold: ParameterManifold | None
    calibration_scores: dict[str, np.ndarray]


def raw_scores(query_z, support: ModelSupport):
    output = pairwise_support_scores(query_z, support.reference_z)
    output["gaussian_nll"] = gaussian_nll(query_z, support.gaussian)
    output["param_manifold"] = parameter_manifold_distance(
        query_z, support.parameter_manifold
    )
    return {name: output[name] for name in SCORE_KINDS}


def compatibility_from_calibration(calibration_scores, query_scores):
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    query_scores = np.asarray(query_scores, dtype=np.float64)
    calibration_scores = calibration_scores[np.isfinite(calibration_scores)]
    result = np.full(len(query_scores), np.nan, dtype=np.float64)
    if len(calibration_scores) == 0:
        return result
    for i, value in enumerate(query_scores):
        if np.isfinite(value):
            result[i] = (
                1.0 + np.sum(calibration_scores >= value)
            ) / (len(calibration_scores) + 1.0)
    return result


def build_supports(latent, experiment, candidate_models):
    supports = {}
    diagnostics = []
    for model_name in require_models(candidate_models):
        ref_idx = indices_for_data_split(experiment, model_name, "train")
        cal_idx = indices_for_data_split(experiment, model_name, "validation")
        if len(ref_idx) < 2 or len(cal_idx) < 1:
            raise ValueError(
                f"Need reference and calibration subjects for {model_name}."
            )
        z_ref = latent[ref_idx]
        z_cal = latent[cal_idx]
        theta_ref = theta_for_combined_indices(experiment, model_name, ref_idx)
        theta_cal = theta_for_combined_indices(experiment, model_name, cal_idx)
        manifold = fit_parameter_manifold(
            theta_ref,
            z_ref,
            theta_val=theta_cal,
            z_val=z_cal,
        )
        gaussian = fit_gaussian_support(z_ref)
        provisional = ModelSupport(
            model_name=model_name,
            reference_z=z_ref,
            gaussian=gaussian,
            parameter_manifold=manifold,
            calibration_scores={},
        )
        cal_scores = raw_scores(z_cal, provisional)
        support = ModelSupport(
            model_name=model_name,
            reference_z=z_ref,
            gaussian=gaussian,
            parameter_manifold=manifold,
            calibration_scores=cal_scores,
        )
        supports[model_name] = support
        diagnostics.append(
            {
                "model_name": model_name,
                "new_name": model_new_name(model_name),
                "family": model_family(model_name),
                "n_reference": len(ref_idx),
                "n_calibration": len(cal_idx),
                "param_manifold_train_r2": (
                    np.nan if manifold is None else manifold.train_r2
                ),
                "param_manifold_validation_r2": (
                    np.nan if manifold is None else manifold.validation_r2
                ),
            }
        )
    return supports, pd.DataFrame(diagnostics)


def score_query_set(
    latent,
    experiment,
    supports,
    query_data_name,
    query_indices,
    representation_label_value,
    protocol,
):
    if len(query_indices) == 0:
        return pd.DataFrame()
    z_query = latent[query_indices]
    src = source_indices(experiment, query_indices)
    ids_all = load_subject_ids(TRIAL_DATA_NAME, query_data_name)
    subject_ids = ids_all[src]
    rows = []
    for model_name, support in supports.items():
        values_by_score = raw_scores(z_query, support)
        for score_name, values in values_by_score.items():
            compatibility = compatibility_from_calibration(
                support.calibration_scores[score_name], values
            )
            for local_index in range(len(query_indices)):
                rows.append(
                    {
                        "protocol": protocol,
                        "representation": representation_label_value,
                        "query_data_name": query_data_name,
                        "query_true_model": (
                            query_data_name if query_data_name.startswith("m") else ""
                        ),
                        "query_true_family": (
                            model_family(query_data_name)
                            if query_data_name.startswith("m")
                            else np.nan
                        ),
                        "combined_subject_index": int(query_indices[local_index]),
                        "source_subject_index": int(src[local_index]),
                        "subject_id": str(subject_ids[local_index]),
                        "candidate_model": model_name,
                        "candidate_new_name": model_new_name(model_name),
                        "candidate_family": model_family(model_name),
                        "score": score_name,
                        "raw_score": float(values[local_index]),
                        "compatibility": float(compatibility[local_index]),
                    }
                )
    return pd.DataFrame(rows)


def best_from_long(score_table: pd.DataFrame):
    if score_table.empty:
        return pd.DataFrame()
    keys = [
        "protocol",
        "representation",
        "query_data_name",
        "query_true_model",
        "query_true_family",
        "combined_subject_index",
        "source_subject_index",
        "subject_id",
        "score",
    ]
    rows = []
    for key, group in score_table.groupby(keys, dropna=False, sort=False):
        finite = group[np.isfinite(group["compatibility"])].copy()
        if finite.empty:
            continue
        best_row = finite.loc[finite["compatibility"].idxmax()]
        family_compat = finite.groupby("candidate_family")["compatibility"].max()
        best_family = int(family_compat.idxmax())
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "best_model": best_row["candidate_model"],
                "best_new_name": best_row["candidate_new_name"],
                "best_family": best_family,
                "c_max": float(best_row["compatibility"]),
            }
        )
        for threshold in COMPATIBILITY_THRESHOLDS:
            row[f"outside_all_at_{threshold:g}"] = bool(
                best_row["compatibility"] < threshold
            )
        rows.append(row)
    return pd.DataFrame(rows)







def find_fit_metrics_file(query_name: str, model_name: str, required_sources):
    root = (
        PROJECT_ROOT
        / "output"
        / "fit_lik"
        / TRIAL_DATA_NAME
        / query_name
        / model_name
    )
    candidates = sorted(root.glob(f"n*_s{SEED}/subject_metrics.csv"))
    matches = []
    required_sources = set(int(x) for x in required_sources)
    for path in candidates:
        frame = pd.read_csv(path)
        available = set(frame["source_subject_index"].astype(int).tolist())
        if required_sources <= available:
            matches.append((len(available), path, frame))
    if not matches:
        return None, None
    matches.sort(key=lambda item: (item[0], str(item[1])))
    return matches[0][1], matches[0][2]


def load_bic_long(experiment, query_names, candidate_models):
    rows = []
    for query_name in query_names:
        if query_name.startswith("r"):
            query_idx = all_indices_for_data(experiment, query_name)
        else:
            query_idx = indices_for_data_split(experiment, query_name, "test")
        src = source_indices(experiment, query_idx)
        for model_name in candidate_models:
            path, frame = find_fit_metrics_file(query_name, model_name, src)
            if path is None:
                log(
                    f"[skip BIC] missing fit covering {query_name} x {model_name}"
                )
                continue
            selected = frame[
                frame["source_subject_index"].astype(int).isin(src)
            ].copy()
            selected["query_data_name"] = query_name
            selected["candidate_model"] = model_name
            selected = selected.rename(columns={"bic": "bic"})
            rows.append(
                selected[
                    [
                        "query_data_name",
                        "source_subject_index",
                        "subject_id",
                        "candidate_model",
                        "bic",
                        "log_likelihood",
                    ]
                ]
            )
            log(f"Loaded BIC: {path}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def classifier_probability_long(
    experiment,
    rep: LoadedRepresentation,
    probabilities,
    query_names,
    candidate_models,
):
    if probabilities is None or rep.classifier_model_names is None:
        return pd.DataFrame()
    name_to_col = {
        name: i for i, name in enumerate(rep.classifier_model_names)
    }
    if not set(candidate_models) <= set(name_to_col):
        return pd.DataFrame()
    rows = []
    for query_name in query_names:
        query_idx = (
            all_indices_for_data(experiment, query_name)
            if query_name.startswith("r")
            else indices_for_data_split(experiment, query_name, "test")
        )
        src = source_indices(experiment, query_idx)
        ids_all = load_subject_ids(TRIAL_DATA_NAME, query_name)
        for model_name in candidate_models:
            values = probabilities[query_idx, name_to_col[model_name]]
            for i in range(len(query_idx)):
                rows.append(
                    {
                        "query_data_name": query_name,
                        "source_subject_index": int(src[i]),
                        "subject_id": str(ids_all[src[i]]),
                        "candidate_model": model_name,
                        "classifier_prob": float(values[i]),
                    }
                )
    return pd.DataFrame(rows)


def winner_table_from_metric(long_table, value_col, higher_is_better=True):
    if long_table.empty:
        return pd.DataFrame()
    rows = []
    keys = ["query_data_name", "source_subject_index", "subject_id"]
    for key, group in long_table.groupby(keys, sort=False):
        values = group[value_col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        valid_indices = np.flatnonzero(finite)
        if higher_is_better:
            selected = valid_indices[np.argmax(values[finite])]
        else:
            selected = valid_indices[np.argmin(values[finite])]
        rows.append(
            {
                "query_data_name": key[0],
                "source_subject_index": int(key[1]),
                "subject_id": str(key[2]),
                "winner": str(group.iloc[selected]["candidate_model"]),
            }
        )
    return pd.DataFrame(rows)







def binary_auroc(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    y_true = y_true[finite]
    scores = scores[finite]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores, method="average")
    rank_sum_pos = ranks[y_true == 1].sum()
    return float(
        (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def mean_within_subject_spearman(left, right, key_columns, model_column):
    merged = left.merge(
        right,
        on=key_columns + [model_column],
        how="inner",
        suffixes=("_left", "_right"),
    )
    rhos = []
    for _, group in merged.groupby(key_columns, sort=False):
        x = group["value_left"].to_numpy(dtype=float)
        y = group["value_right"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            continue
        if np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
            continue
        rho, _ = spearmanr(x[finite], y[finite])
        if np.isfinite(rho):
            rhos.append(float(rho))
    return (float(np.mean(rhos)), len(rhos)) if rhos else (np.nan, 0)


def generalization_summary(best):
    if best.empty:
        return pd.DataFrame()
    rows = []
    for (representation, score), group in best.groupby(
        ["representation", "score"], sort=False
    ):
        valid = group[np.isfinite(group["query_true_family"])].copy()
        rows.append(
            {
                "representation": representation,
                "score": score,
                "n_subjects": len(valid),
                "family_accuracy": float(
                    (valid["best_family"] == valid["query_true_family"]).mean()
                ),
                "mean_c_max": float(valid["c_max"].mean()),
            }
        )
        for query_model, model_group in valid.groupby("query_data_name"):
            rows.append(
                {
                    "representation": representation,
                    "score": f"{score}::{query_model}",
                    "n_subjects": len(model_group),
                    "family_accuracy": float(
                        (
                            model_group["best_family"]
                            == model_group["query_true_family"]
                        ).mean()
                    ),
                    "mean_c_max": float(model_group["c_max"].mean()),
                }
            )
    return pd.DataFrame(rows)


def recovery_summary(best):
    if best.empty:
        return pd.DataFrame()
    rows = []
    for (representation, score), group in best.groupby(
        ["representation", "score"], sort=False
    ):
        valid = group[group["query_true_model"] != ""].copy()
        rows.append(
            {
                "method": f"{representation} | {score}",
                "representation": representation,
                "score": score,
                "n_subjects": len(valid),
                "exact_model_accuracy": float(
                    (valid["best_model"] == valid["query_true_model"]).mean()
                ),
                "family_accuracy": float(
                    (valid["best_family"] == valid["query_true_family"]).mean()
                ),
                "mean_c_max": float(valid["c_max"].mean()),
            }
        )
    return pd.DataFrame(rows)


def auxiliary_recovery_rows(bic_long, clf_long):
    rows = []
    winner_frames = {}
    if not bic_long.empty:
        bic_w = winner_table_from_metric(bic_long, "bic", higher_is_better=False)
        winner_frames["BIC"] = bic_w
    if not clf_long.empty:
        clf_w = winner_table_from_metric(
            clf_long, "classifier_prob", higher_is_better=True
        )
        winner_frames["Classifier output"] = clf_w
    for method, winners in winner_frames.items():
        if winners.empty:
            continue
        true = winners["query_data_name"]
        exact = winners["winner"] == true
        family = np.array([model_family(m) for m in winners["winner"]]) == np.array(
            [model_family(m) for m in true]
        )
        rows.append(
            {
                "method": method,
                "representation": "",
                "score": "",
                "n_subjects": len(winners),
                "exact_model_accuracy": float(exact.mean()),
                "family_accuracy": float(family.mean()),
                "mean_c_max": np.nan,
            }
        )
    return pd.DataFrame(rows), winner_frames


def recovery_agreement(best, bic_long, clf_long, winner_frames):
    if best.empty:
        return pd.DataFrame()
    rows = []
    key = ["query_data_name", "source_subject_index"]
    for (representation, score), group in best.groupby(
        ["representation", "score"], sort=False
    ):
        latent_w = group[key + ["best_model"]].rename(
            columns={"best_model": "latent_winner"}
        )
        for method, winners in winner_frames.items():
            merged = latent_w.merge(
                winners[key + ["winner"]], on=key, how="inner"
            )
            rows.append(
                {
                    "representation": representation,
                    "score": score,
                    "comparison": method,
                    "n_subjects": len(merged),
                    "winner_agreement": (
                        float((merged["latent_winner"] == merged["winner"]).mean())
                        if len(merged)
                        else np.nan
                    ),
                    "mean_within_subject_spearman": np.nan,
                }
            )



        latent_values = CURRENT_RECOVERY_SCORE_TABLE[
            (CURRENT_RECOVERY_SCORE_TABLE["representation"] == representation)
            & (CURRENT_RECOVERY_SCORE_TABLE["score"] == score)
        ][
            [
                "query_data_name",
                "source_subject_index",
                "candidate_model",
                "compatibility",
            ]
        ].rename(columns={"compatibility": "value"})
        if not bic_long.empty:
            bic_values = bic_long[
                ["query_data_name", "source_subject_index", "candidate_model", "bic"]
            ].copy()
            bic_values["value"] = -bic_values["bic"].astype(float)
            rho, n = mean_within_subject_spearman(
                latent_values,
                bic_values[["query_data_name", "source_subject_index", "candidate_model", "value"]],
                key,
                "candidate_model",
            )
            rows.append(
                {
                    "representation": representation,
                    "score": score,
                    "comparison": "BIC rank evidence",
                    "n_subjects": n,
                    "winner_agreement": np.nan,
                    "mean_within_subject_spearman": rho,
                }
            )
        if not clf_long.empty:
            clf_values = clf_long[
                [
                    "query_data_name",
                    "source_subject_index",
                    "candidate_model",
                    "classifier_prob",
                ]
            ].rename(columns={"classifier_prob": "value"})
            rho, n = mean_within_subject_spearman(
                latent_values,
                clf_values,
                key,
                "candidate_model",
            )
            rows.append(
                {
                    "representation": representation,
                    "score": score,
                    "comparison": "Classifier rank evidence",
                    "n_subjects": n,
                    "winner_agreement": np.nan,
                    "mean_within_subject_spearman": rho,
                }
            )
    return pd.DataFrame(rows)


def misspec_summary(best_with_groups):
    if best_with_groups.empty:
        return pd.DataFrame()
    rows = []
    for (representation, score), group in best_with_groups.groupby(
        ["representation", "score"], sort=False
    ):
        group_rows = {
            label: subset for label, subset in group.groupby("coverage_group")
        }
        for label, subset in group_rows.items():
            rows.append(
                {
                    "representation": representation,
                    "score": score,
                    "statistic": f"group::{label}",
                    "n": len(subset),
                    "value": float(subset["c_max"].mean()),
                    "median": float(subset["c_max"].median()),
                }
            )
        if "different_family_unseen" in group_rows:
            ood = group_rows["different_family_unseen"].copy()
            for negative_label in ("id", "same_family_unseen"):
                if negative_label not in group_rows:
                    continue
                neg = group_rows[negative_label].copy()
                y = np.r_[np.zeros(len(neg), dtype=int), np.ones(len(ood), dtype=int)]
                detection = np.r_[1.0 - neg["c_max"], 1.0 - ood["c_max"]]
                rows.append(
                    {
                        "representation": representation,
                        "score": score,
                        "statistic": f"AUROC::{negative_label}_vs_different_family",
                        "n": len(y),
                        "value": binary_auroc(y, detection),
                        "median": np.nan,
                    }
                )
            covered_parts = [
                group_rows[name]
                for name in ("id", "same_family_unseen")
                if name in group_rows
            ]
            if covered_parts:
                covered = pd.concat(covered_parts, ignore_index=True)
                y = np.r_[
                    np.zeros(len(covered), dtype=int),
                    np.ones(len(ood), dtype=int),
                ]
                detection = np.r_[
                    1.0 - covered["c_max"],
                    1.0 - ood["c_max"],
                ]
                rows.append(
                    {
                        "representation": representation,
                        "score": score,
                        "statistic": "AUROC::covered_family_vs_different_family",
                        "n": len(y),
                        "value": binary_auroc(y, detection),
                        "median": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def real_summary(best):
    if best.empty:
        return pd.DataFrame()
    rows = []
    for (representation, score), group in best.groupby(
        ["representation", "score"], sort=False
    ):
        row = {
            "representation": representation,
            "score": score,
            "n_subjects": len(group),
            "mean_c_max": float(group["c_max"].mean()),
            "median_c_max": float(group["c_max"].median()),
        }
        for threshold in COMPATIBILITY_THRESHOLDS:
            row[f"prop_outside_all_at_{threshold:g}"] = float(
                (group["c_max"] < threshold).mean()
            )
        for model_name in RECOVERY_CANDIDATE_MODELS:
            row[f"prop_best_{model_name}"] = float(
                (group["best_model"] == model_name).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)




CURRENT_RECOVERY_SCORE_TABLE = pd.DataFrame()







def save_figure(fig, name, figures, outdir):
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    figures.append(fig)
    return path


def plot_generalization(summary):
    frame = summary[~summary["score"].str.contains("::", regex=False)].copy()
    if frame.empty:
        return None
    labels = [f"{r}\n{s}" for r, s in zip(frame.representation, frame.score)]
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(frame)), 5.2))
    ax.bar(np.arange(len(frame)), frame["family_accuracy"].to_numpy())
    ax.set_xticks(np.arange(len(frame)), labels=labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Held-out family recovery accuracy")
    ax.set_title("A. Structural generalization to held-out models")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_recovery(summary):
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(summary)), 5.2))
    x = np.arange(len(summary))
    width = 0.38
    ax.bar(x - width / 2, summary["exact_model_accuracy"], width, label="Exact model")
    ax.bar(x + width / 2, summary["family_accuracy"], width, label="Family")
    ax.set_xticks(x, labels=summary["method"], rotation=50, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Recovery accuracy")
    ax.set_title("B. Model recovery")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_misspec(best):
    if best.empty:
        return None
    frame = best[best["score"] == PRIMARY_SCORE].copy()
    if frame.empty:
        return None
    reps = list(dict.fromkeys(frame["representation"].tolist()))
    groups = ["id", "same_family_unseen", "different_family_unseen"]
    fig, axes = plt.subplots(
        1,
        len(reps),
        figsize=(4.5 * len(reps), 4.8),
        squeeze=False,
    )
    for ax, rep in zip(axes[0], reps, strict=True):
        subset = frame[frame["representation"] == rep]
        values = [
            subset.loc[subset["coverage_group"] == group, "c_max"].to_numpy()
            for group in groups
        ]
        ax.boxplot(values, labels=["ID", "unseen\nsame family", "unseen\nother family"])
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("C_max" if ax is axes[0][0] else "")
        ax.set_title(rep)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.suptitle(f"C. Candidate-family coverage ({PRIMARY_SCORE})")
    fig.tight_layout()
    return fig


def plot_real(best):
    if best.empty:
        return None
    frame = best[best["score"] == PRIMARY_SCORE].copy()
    if frame.empty:
        return None
    reps = list(dict.fromkeys(frame["representation"].tolist()))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    values = [
        frame.loc[frame["representation"] == rep, "c_max"].to_numpy()
        for rep in reps
    ]
    ax.boxplot(values, labels=reps)
    for threshold in COMPATIBILITY_THRESHOLDS:
        ax.axhline(threshold, linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("C_max")
    ax.set_title(f"D. Real-subject simulator coverage ({PRIMARY_SCORE})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig







def encode_protocol_representations(experiment, train_models):
    outputs = {}
    for kind in REPRESENTATION_KINDS:
        rep = load_representation(kind, train_models)
        if rep is None:
            continue
        latent_raw, probabilities = encode_representation(rep, experiment)
        scaler = fit_latent_scale(latent_raw, experiment, train_models)
        latent = scaler.transform(latent_raw)
        outputs[rep.label] = {
            "rep": rep,
            "latent": latent,
            "probabilities": probabilities,
            "latent_scale_mean": scaler.mean,
            "latent_scale_sd": scaler.scale,
        }
    return outputs


def run_generalization(experiment, rep_outputs):
    score_tables = []
    diagnostics = []
    for rep_label, info in rep_outputs.items():
        supports, diag = build_supports(
            info["latent"], experiment, GENERALIZATION_TRAIN_MODELS
        )
        diag.insert(0, "representation", rep_label)
        diag.insert(0, "protocol", "generalization")
        diagnostics.append(diag)
        for query_name in GENERALIZATION_HELDOUT_MODELS:
            query_idx = indices_for_data_split(experiment, query_name, "test")
            score_tables.append(
                score_query_set(
                    info["latent"],
                    experiment,
                    supports,
                    query_name,
                    query_idx,
                    rep_label,
                    "generalization",
                )
            )
    scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    best = best_from_long(scores)
    summary = generalization_summary(best)
    diagnostics_frame = (
        pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    )
    return scores, best, summary, diagnostics_frame


def run_recovery(experiment, rep_outputs):
    global CURRENT_RECOVERY_SCORE_TABLE
    score_tables = []
    diagnostics = []
    classifier_long = pd.DataFrame()
    for rep_label, info in rep_outputs.items():
        supports, diag = build_supports(
            info["latent"], experiment, RECOVERY_CANDIDATE_MODELS
        )
        diag.insert(0, "representation", rep_label)
        diag.insert(0, "protocol", "recovery")
        diagnostics.append(diag)
        for query_name in RECOVERY_QUERY_MODELS:
            query_idx = indices_for_data_split(experiment, query_name, "test")
            score_tables.append(
                score_query_set(
                    info["latent"],
                    experiment,
                    supports,
                    query_name,
                    query_idx,
                    rep_label,
                    "recovery",
                )
            )
        if info["rep"].kind == "classifier":
            classifier_long = classifier_probability_long(
                experiment,
                info["rep"],
                info["probabilities"],
                RECOVERY_QUERY_MODELS,
                RECOVERY_CANDIDATE_MODELS,
            )
    scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    CURRENT_RECOVERY_SCORE_TABLE = scores
    best = best_from_long(scores)
    latent_summary = recovery_summary(best)
    bic_long = load_bic_long(
        experiment, RECOVERY_QUERY_MODELS, RECOVERY_CANDIDATE_MODELS
    )
    aux_summary, winner_frames = auxiliary_recovery_rows(bic_long, classifier_long)
    summary = pd.concat([latent_summary, aux_summary], ignore_index=True)
    agreement = recovery_agreement(best, bic_long, classifier_long, winner_frames)
    diagnostics_frame = (
        pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    )
    return (
        scores,
        best,
        summary,
        agreement,
        bic_long,
        classifier_long,
        diagnostics_frame,
    )


def run_misspec(experiment, rep_outputs):
    score_tables = []
    diagnostics = []
    group_map = {
        **{m: "id" for m in MISSPEC_ID_MODELS},
        **{m: "same_family_unseen" for m in MISSPEC_SAME_FAMILY_UNSEEN},
        **{m: "different_family_unseen" for m in MISSPEC_OUT_FAMILY_UNSEEN},
    }
    for rep_label, info in rep_outputs.items():
        supports, diag = build_supports(
            info["latent"], experiment, MISSPEC_CANDIDATE_MODELS
        )
        diag.insert(0, "representation", rep_label)
        diag.insert(0, "protocol", "misspecification")
        diagnostics.append(diag)
        for query_name, group_label in group_map.items():
            query_idx = indices_for_data_split(experiment, query_name, "test")
            table = score_query_set(
                info["latent"],
                experiment,
                supports,
                query_name,
                query_idx,
                rep_label,
                "misspecification",
            )
            table["coverage_group"] = group_label
            score_tables.append(table)
    scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    best = best_from_long(scores)
    if not best.empty:
        best["coverage_group"] = best["query_data_name"].map(group_map)
    summary = misspec_summary(best)
    diagnostics_frame = (
        pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    )
    return scores, best, summary, diagnostics_frame


def run_real(experiment, rep_outputs):
    score_tables = []
    diagnostics = []
    classifier_long = pd.DataFrame()
    for rep_label, info in rep_outputs.items():
        supports, diag = build_supports(
            info["latent"], experiment, RECOVERY_CANDIDATE_MODELS
        )
        diag.insert(0, "representation", rep_label)
        diag.insert(0, "protocol", "real")
        diagnostics.append(diag)
        for query_name in REAL_DATA_NAMES:
            query_idx = all_indices_for_data(experiment, query_name)
            score_tables.append(
                score_query_set(
                    info["latent"],
                    experiment,
                    supports,
                    query_name,
                    query_idx,
                    rep_label,
                    "real",
                )
            )
        if info["rep"].kind == "classifier":
            classifier_long = classifier_probability_long(
                experiment,
                info["rep"],
                info["probabilities"],
                REAL_DATA_NAMES,
                RECOVERY_CANDIDATE_MODELS,
            )
    scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    best = best_from_long(scores)
    summary = real_summary(best)
    bic_long = load_bic_long(
        experiment, REAL_DATA_NAMES, RECOVERY_CANDIDATE_MODELS
    )
    diagnostics_frame = (
        pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    )
    return scores, best, summary, bic_long, classifier_long, diagnostics_frame







def main():
    outdir = output_dir()
    log(f"Device: {DEVICE}")
    log(f"Evaluation trial data: {TRIAL_DATA_NAME}")
    log(f"Real data: {REAL_DATA_NAMES}")
    log(f"Models: {MODEL_DATA_NAMES}")
    log(f"Scores: {SCORE_KINDS}")
    log(f"Primary score: {PRIMARY_SCORE}")
    log(f"Latent-axis standardization: {STANDARDIZE_LATENT_AXES}")
    log(
        "Design matching: real and simulated comparison data are loaded from "
        f"the same trial dataset ({TRIAL_DATA_NAME})."
    )

    all_data_names = order_data_names((*REAL_DATA_NAMES, *MODEL_DATA_NAMES))
    experiment = load_experiment_data(
        TRIAL_DATA_NAME,
        all_data_names,
        seed=SEED,
        full_real=True,
    )


    log("\n=== Loading generalization representations (trained m4--m8) ===")
    generalization_reps = encode_protocol_representations(
        experiment, GENERALIZATION_TRAIN_MODELS
    )


    log("\n=== Loading recovery representations (trained m1--m3) ===")
    recovery_reps = encode_protocol_representations(
        experiment, RECOVERY_TRAIN_MODELS
    )

    log("\n=== A. Held-out structural generalization ===")
    gen_scores, gen_best, gen_summary, gen_diag = run_generalization(
        experiment, generalization_reps
    )
    log(gen_summary.to_string(index=False) if not gen_summary.empty else "(no results)")

    log("\n=== B. Model recovery ===")
    (
        rec_scores,
        rec_best,
        rec_summary,
        rec_agreement,
        rec_bic,
        rec_clf,
        rec_diag,
    ) = run_recovery(experiment, recovery_reps)
    log(rec_summary.to_string(index=False) if not rec_summary.empty else "(no results)")
    if not rec_agreement.empty:
        log("\nRecovery agreement:")
        log(rec_agreement.to_string(index=False))

    log("\n=== C. Candidate-family misspecification / coverage ===")
    mis_scores, mis_best, mis_summary, mis_diag = run_misspec(
        experiment, generalization_reps
    )
    log(mis_summary.to_string(index=False) if not mis_summary.empty else "(no results)")

    log("\n=== D. Real-subject application ===")
    real_scores, real_best, real_summary_frame, real_bic, real_clf, real_diag = run_real(
        experiment, recovery_reps
    )
    log(
        real_summary_frame.to_string(index=False)
        if not real_summary_frame.empty
        else "(no results)"
    )


    tables = {
        "generalization_scores_long.csv": gen_scores,
        "generalization_subjects.csv": gen_best,
        "generalization_summary.csv": gen_summary,
        "recovery_scores_long.csv": rec_scores,
        "recovery_subjects.csv": rec_best,
        "recovery_summary.csv": rec_summary,
        "recovery_agreement.csv": rec_agreement,
        "recovery_bic_long.csv": rec_bic,
        "recovery_classifier_long.csv": rec_clf,
        "misspec_scores_long.csv": mis_scores,
        "misspec_subjects.csv": mis_best,
        "misspec_summary.csv": mis_summary,
        "real_scores_long.csv": real_scores,
        "real_subjects.csv": real_best,
        "real_summary.csv": real_summary_frame,
        "real_bic_long.csv": real_bic,
        "real_classifier_long.csv": real_clf,
        "support_diagnostics.csv": pd.concat(
            [gen_diag, rec_diag, mis_diag, real_diag],
            ignore_index=True,
        )
        if any(not frame.empty for frame in (gen_diag, rec_diag, mis_diag, real_diag))
        else pd.DataFrame(),
    }
    for filename, frame in tables.items():
        if frame is not None and not frame.empty:
            frame.to_csv(outdir / filename, index=False)


    config = {
        "trial_data_name": TRIAL_DATA_NAME,
        "real_data_names": list(REAL_DATA_NAMES),
        "model_data_names": list(MODEL_DATA_NAMES),
        "representation_train_trial_data_name": REPRESENTATION_TRAIN_TRIAL_DATA_NAME,
        "generalization_train_models": list(GENERALIZATION_TRAIN_MODELS),
        "generalization_heldout_models": list(GENERALIZATION_HELDOUT_MODELS),
        "recovery_train_models": list(RECOVERY_TRAIN_MODELS),
        "recovery_candidate_models": list(RECOVERY_CANDIDATE_MODELS),
        "misspec_candidate_models": list(MISSPEC_CANDIDATE_MODELS),
        "misspec_id_models": list(MISSPEC_ID_MODELS),
        "misspec_same_family_unseen": list(MISSPEC_SAME_FAMILY_UNSEEN),
        "misspec_out_family_unseen": list(MISSPEC_OUT_FAMILY_UNSEEN),
        "representations_requested": list(REPRESENTATION_KINDS),
        "scores": list(SCORE_KINDS),
        "primary_score": PRIMARY_SCORE,
        "knn_k": KNN_K,
        "standardize_latent_axes": STANDARDIZE_LATENT_AXES,
        "gaussian_ridge": GAUSSIAN_RIDGE,
        "param_manifold_poly_degree": PARAM_MANIFOLD_POLY_DEGREE,
        "param_manifold_ridge": PARAM_MANIFOLD_RIDGE,
        "param_manifold_continuous_refine": PARAM_MANIFOLD_CONTINUOUS_REFINE,
        "max_points_per_data": MAX_POINTS_PER_DATA,
        "seed": SEED,
        "device": DEVICE,
    }
    (outdir / "analysis_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


    figures = []
    figure_paths = []
    for name, figure in (
        ("fig_A_generalization", plot_generalization(gen_summary)),
        ("fig_B_recovery", plot_recovery(rec_summary)),
        ("fig_C_misspecification", plot_misspec(mis_best)),
        ("fig_D_real_coverage", plot_real(real_best)),
    ):
        if figure is not None:
            figure_paths.append(save_figure(figure, name, figures, outdir))
    if figures:
        with PdfPages(outdir / "figures.pdf") as pdf:
            for figure in figures:
                pdf.savefig(figure, bbox_inches="tight")

    log("\n=== Saved outputs ===")
    for path in sorted(outdir.iterdir()):
        if path.is_file():
            log(path)
    (outdir / "results.txt").write_text(
        "\n".join(REPORT_LINES) + "\n", encoding="utf-8"
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)

    print(f"\nFinished. Outputs: {outdir}")


if __name__ == "__main__":

    warnings.simplefilter("default")
    main()
