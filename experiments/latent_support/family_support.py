from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import math
import sys
import traceback
import warnings

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
CORE_PATH = THIS_DIR / "support_core.py"
if not CORE_PATH.exists():
    raise FileNotFoundError(
        f"Expected the previous analysis file next to this script: {CORE_PATH}"
    )

spec = importlib.util.spec_from_file_location("latent_support_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not import {CORE_PATH}")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


TRIAL_DATA_NAMES = (
    "pub_Konstantinidis2020",
    "pub_Thrailkill2023",
)
PRIMARY_TRIAL_DATA_NAME = "pub_Konstantinidis2020"

PRIMARY_SPLIT_SEED = 31
SPLIT_ROBUSTNESS_SEEDS = (31, 32, 33)

ALL_REPRESENTATIONS = (
    "nn1",
    "nn1b",
    "classifier",
    "nn1c",
    "nn1bc",
    "parameter_decoder",
)


PRIMARY_SCORE_KINDS = (
    "mean_euclidean",
    "knn",
    "gaussian_nll",
)
PRIMARY_K = 10
KNN_K_VALUES = (5, 10, 20, 40)


GENERALIZATION_TRAIN_MODELS = ("m4", "m5", "m6", "m7", "m8")
GENERALIZATION_HELDOUT_MODELS = ("m1", "m2", "m3")


M4M8_PARALLEL_TRAIN_MODELS = GENERALIZATION_TRAIN_MODELS
M4M8_PARALLEL_CANDIDATE_MODELS = ("m1", "m2", "m3")


M4M8_TRAINING_FAMILY_MODELS = {
    "option_wise": ("m4", "m5", "m6", "m7"),
    "attribute_wise": ("m8",),
}
M4M8_TRAINING_FAMILY_LABELS = {
    "option_wise": "Option-wise value",
    "attribute_wise": "Attribute-wise comparison",
}
M4M8_HELDOUT_EXPECTED_FAMILY = {
    "m1": "option_wise",
    "m2": "option_wise",
    "m3": "attribute_wise",
}

FAMILY_COVERAGE_CANDIDATES = ("m4", "m5", "m6")
FAMILY_COVERAGE_ID = ("m4", "m5", "m6")
FAMILY_COVERAGE_SAME_FAMILY_UNSEEN = ("m1", "m2")
FAMILY_COVERAGE_DIFFERENT_FAMILY_UNSEEN = ("m3",)


CANDIDATE_OOD_CANDIDATES = ("m1", "m2", "m3")
CANDIDATE_OOD_ID = ("m1", "m2", "m3")
CANDIDATE_OOD_RELATED_UNSEEN = ("m4", "m5", "m6", "m8")
CANDIDATE_OOD_NOVEL_FAMILY = ("m7",)


APPLICATION_QUERY_NAMES = ("r1", "m9")
APPLICATION_SIMULATED_N = 1000
APPLICATION_SAMPLE_SEED = 31

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 260823
CI_LEVEL = 0.95

MAX_POINTS_PER_DATA = 1000
SHOW_FIGURES = False
SAVE_LONG_TABLES_FOR_PRIMARY = True


CFG_DIR = THIS_DIR / "configs"
ACTIVE_CONFIG = "cfg4"


DISABLE_PARAMETER_MANIFOLD = True


REPORT_LINES: list[str] = []
RUN_MANIFEST_ROWS: list[dict] = []
CHECKPOINT_ROWS: list[dict] = []


def load_representation_config(config_name: str) -> dict:
    path = CFG_DIR / f"{config_name}.py"
    if not path.exists():
        available = sorted(p.stem for p in CFG_DIR.glob("cfg*.py"))
        raise FileNotFoundError(
            f"Missing config file: {path}. Available: {available}"
        )
    spec = importlib.util.spec_from_file_location(
        f"latent_support_cfg_{config_name}",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hparams = getattr(module, "REPRESENTATION_HPARAMS", None)
    if not isinstance(hparams, dict):
        raise ValueError(
            f"{path} must define REPRESENTATION_HPARAMS as a dict."
        )
    return {
        "config_name": getattr(module, "CONFIG_NAME", config_name),
        "config_file": str(path),
        "representation_hparams": hparams,
    }


ACTIVE_REPRESENTATION_CONFIG = load_representation_config(ACTIVE_CONFIG)
ANALYSIS_OUTPUT_NAME = "parallel_m4m8"
OUTPUT_ROOT = (
    core.PROJECT_ROOT
    / "output"
    / "latent_support_final_analysis"
    / ANALYSIS_OUTPUT_NAME
    / ACTIVE_REPRESENTATION_CONFIG["config_name"]
)


def log(message=""):
    text = str(message)
    print(text)
    REPORT_LINES.append(text)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_frame(frame: pd.DataFrame | None, path: Path):
    if frame is None or frame.empty:
        return
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def add_metadata(frame: pd.DataFrame | None, **metadata) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    for key, value in reversed(list(metadata.items())):
        result.insert(0, key, value)
    return result


def trial_data_available(trial_data_name: str) -> tuple[bool, str]:


    required = ("r1",) + tuple(f"m{i}" for i in range(1, 9))
    missing = []
    for name in required:
        try:
            core.choice_data_path(trial_data_name, name)
        except Exception:
            missing.append(name)
    if missing:
        return False, f"missing choice datasets: {missing}"
    return True, "ok"


ACTIVE_APPLICATION_QUERIES: tuple[str, ...] = tuple(APPLICATION_QUERY_NAMES)


def application_query_names() -> tuple[str, ...]:
    names = tuple(dict.fromkeys(ACTIVE_APPLICATION_QUERIES))
    if not names:
        raise ValueError("APPLICATION_QUERY_NAMES must be non-empty.")
    return names


def configured_application_query_names() -> tuple[str, ...]:
    names = tuple(dict.fromkeys(APPLICATION_QUERY_NAMES))
    if not names:
        raise ValueError("APPLICATION_QUERY_NAMES must be non-empty.")
    return names


def available_application_query_names(trial_data_name: str) -> tuple[str, ...]:
    kept = []
    skipped = []
    for name in configured_application_query_names():
        try:
            core.choice_data_path(trial_data_name, name)
        except Exception:
            skipped.append(name)
            continue
        kept.append(name)
    if skipped:
        log(
            f"[skip application query] {trial_data_name}: missing {skipped}"
        )
    if not kept:
        raise ValueError(
            f"No APPLICATION_QUERY_NAMES available for {trial_data_name}."
        )
    return tuple(kept)

def application_query_indices(experiment, query_name: str) -> np.ndarray:


    all_idx = (
        experiment.subject_indices_by_data[query_name]
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    if query_name.startswith("r"):
        return core.limit_indices(all_idx)
    if not query_name.startswith("m"):
        raise ValueError(
            f"Unsupported application query name: {query_name!r} "
            "(expected r* or m*)."
        )
    n_take = int(APPLICATION_SIMULATED_N)
    if n_take <= 0:
        raise ValueError("APPLICATION_SIMULATED_N must be positive.")
    if len(all_idx) <= n_take:
        return all_idx
    rng = np.random.default_rng(APPLICATION_SAMPLE_SEED)
    chosen = rng.choice(len(all_idx), size=n_take, replace=False)
    return np.sort(all_idx[chosen])


def application_data_names_to_load() -> tuple[str, ...]:
    unique = tuple(
        dict.fromkeys(
            (
                *core.REAL_DATA_NAMES,
                *core.MODEL_DATA_NAMES,
                *application_query_names(),
            )
        )
    )
    return core.order_data_names(unique)

def configure_core(trial_data_name: str, split_seed: int, knn_k: int):
    core.TRIAL_DATA_NAME = trial_data_name
    core.SEED = int(split_seed)
    core.KNN_K = int(knn_k)
    core.REPRESENTATION_KINDS = tuple(ALL_REPRESENTATIONS)
    core.REPRESENTATION_HPARAMS = dict(
        ACTIVE_REPRESENTATION_CONFIG["representation_hparams"]
    )
    core.SCORE_KINDS = tuple(PRIMARY_SCORE_KINDS)
    core.PRIMARY_SCORE = "knn"
    core.MAX_POINTS_PER_DATA = MAX_POINTS_PER_DATA
    core.SHOW_FIGURES = False

    core.GENERALIZATION_TRAIN_MODELS = tuple(GENERALIZATION_TRAIN_MODELS)
    core.GENERALIZATION_HELDOUT_MODELS = tuple(GENERALIZATION_HELDOUT_MODELS)
    core.RECOVERY_TRAIN_MODELS = tuple(CANDIDATE_OOD_CANDIDATES)
    core.RECOVERY_QUERY_MODELS = tuple(CANDIDATE_OOD_CANDIDATES)
    core.RECOVERY_CANDIDATE_MODELS = tuple(CANDIDATE_OOD_CANDIDATES)
    core.MISSPEC_CANDIDATE_MODELS = tuple(FAMILY_COVERAGE_CANDIDATES)
    core.MISSPEC_ID_MODELS = tuple(FAMILY_COVERAGE_ID)
    core.MISSPEC_SAME_FAMILY_UNSEEN = tuple(
        FAMILY_COVERAGE_SAME_FAMILY_UNSEEN
    )
    core.MISSPEC_OUT_FAMILY_UNSEEN = tuple(
        FAMILY_COVERAGE_DIFFERENT_FAMILY_UNSEEN
    )

    if DISABLE_PARAMETER_MANIFOLD:


        core.fit_parameter_manifold = lambda *args, **kwargs: None


def checkpoint_coverage(train_models, protocol: str) -> pd.DataFrame:
    rows = []
    for kind in ALL_REPRESENTATIONS:
        path = core.representation_checkpoint_path(kind, train_models)
        hparams = core.representation_hparams_for(kind)
        rows.append(
            {
                "protocol": protocol,
                "train_models": ",".join(train_models),
                "representation_kind": kind,
                "representation": core.representation_label(kind),
                "beta": hparams["beta"],
                "consistency_weight": hparams["consistency_weight"],
                "classification_weight": hparams["classification_weight"],
                "checkpoint_path": str(path),
                "checkpoint_exists": bool(path.exists()),
            }
        )
    frame = pd.DataFrame(rows)
    CHECKPOINT_ROWS.extend(frame.to_dict("records"))
    return frame


def run_recovery_latent_only(experiment, rep_outputs):
    tables = []
    for rep_label, info in rep_outputs.items():
        supports, _ = core.build_supports(
            info["latent"], experiment, CANDIDATE_OOD_CANDIDATES
        )
        for query_name in CANDIDATE_OOD_ID:
            query_idx = core.indices_for_data_split(
                experiment, query_name, "test"
            )
            tables.append(
                core.score_query_set(
                    info["latent"],
                    experiment,
                    supports,
                    query_name,
                    query_idx,
                    rep_label,
                    "recovery",
                )
            )
    scores = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    best = core.best_from_long(scores)
    summary = core.recovery_summary(best)
    return scores, best, summary


def application_summary(best):
    if best is None or best.empty:
        return pd.DataFrame()
    group_cols = ["representation", "score", "query_data_name"]
    families = sorted(
        {core.model_family(m) for m in CANDIDATE_OOD_CANDIDATES}
    )
    rows = []
    for keys, group in best.groupby(group_cols, sort=False):
        representation, score, query_name = keys
        row = {
            "query_data_name": query_name,
            "representation": representation,
            "score": score,
            "n_subjects": len(group),
            "mean_c_max": float(group["c_max"].mean()),
            "median_c_max": float(group["c_max"].median()),
        }
        for threshold in core.COMPATIBILITY_THRESHOLDS:
            row[f"prop_outside_all_at_{threshold:g}"] = float(
                (group["c_max"] < threshold).mean()
            )
        for model_name in CANDIDATE_OOD_CANDIDATES:
            row[f"prop_best_{model_name}"] = float(
                (group["best_model"] == model_name).mean()
            )
        best_families = group["best_model"].map(core.model_family)
        for family_id in families:
            row[f"prop_assigned_family_{family_id}"] = float(
                (best_families == family_id).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def classifier_probability_for_application(
    experiment,
    rep,
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
        query_idx = application_query_indices(experiment, query_name)
        src = core.source_indices(experiment, query_idx)
        ids_all = core.load_subject_ids(core.TRIAL_DATA_NAME, query_name)
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


def run_application(experiment, rep_outputs):
    query_names = application_query_names()
    score_tables = []
    diagnostics = []
    classifier_long = pd.DataFrame()
    for rep_label, info in rep_outputs.items():
        supports, diag = core.build_supports(
            info["latent"], experiment, CANDIDATE_OOD_CANDIDATES
        )
        diag.insert(0, "representation", rep_label)
        diag.insert(0, "protocol", "application")
        diagnostics.append(diag)
        for query_name in query_names:
            query_idx = application_query_indices(experiment, query_name)
            log(
                f"  application query {query_name}: "
                f"n={len(query_idx)}"
                + (
                    f" (random subsample of {APPLICATION_SIMULATED_N})"
                    if query_name.startswith("m")
                    else ""
                )
            )
            score_tables.append(
                core.score_query_set(
                    info["latent"],
                    experiment,
                    supports,
                    query_name,
                    query_idx,
                    rep_label,
                    "application",
                )
            )
        if info["rep"].kind == "classifier":
            classifier_long = classifier_probability_for_application(
                experiment,
                info["rep"],
                info["probabilities"],
                query_names,
                CANDIDATE_OOD_CANDIDATES,
            )
    scores = (
        pd.concat(score_tables, ignore_index=True)
        if score_tables
        else pd.DataFrame()
    )
    best = core.best_from_long(scores)
    summary = application_summary(best)
    bic_long = core.load_bic_long(
        experiment, query_names, CANDIDATE_OOD_CANDIDATES
    )
    diagnostics_frame = (
        pd.concat(diagnostics, ignore_index=True)
        if diagnostics
        else pd.DataFrame()
    )
    return scores, best, summary, bic_long, classifier_long, diagnostics_frame


def run_real_latent_only(experiment, rep_outputs):
    tables = []
    for rep_label, info in rep_outputs.items():
        supports, _ = core.build_supports(
            info["latent"], experiment, CANDIDATE_OOD_CANDIDATES
        )
        for query_name in application_query_names():
            query_idx = application_query_indices(experiment, query_name)
            tables.append(
                core.score_query_set(
                    info["latent"],
                    experiment,
                    supports,
                    query_name,
                    query_idx,
                    rep_label,
                    "application",
                )
            )
    scores = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    best = core.best_from_long(scores)
    summary = application_summary(best)
    return scores, best, summary


def tag_m4m8_parallel(frame: pd.DataFrame | None, **extra) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    metadata = {
        "encoder_protocol": "train_m4_m8",
        "encoder_train_models": ",".join(M4M8_PARALLEL_TRAIN_MODELS),
        "candidate_support_models": ",".join(M4M8_PARALLEL_CANDIDATE_MODELS),
    }
    metadata.update(extra)
    return add_metadata(frame, **metadata)


def tag_m4m8_training_family(frame: pd.DataFrame | None, **extra) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    metadata = {
        "encoder_protocol": "train_m4_m8",
        "support_level": "training_family_union",
        "encoder_train_models": ",".join(M4M8_PARALLEL_TRAIN_MODELS),
        "option_wise_support_models": ",".join(
            M4M8_TRAINING_FAMILY_MODELS["option_wise"]
        ),
        "attribute_wise_support_models": ",".join(
            M4M8_TRAINING_FAMILY_MODELS["attribute_wise"]
        ),
    }
    metadata.update(extra)
    return add_metadata(frame, **metadata)


def m4m8_coarse_family_for_model(model_name: str) -> str:
    for family_name, model_names in M4M8_TRAINING_FAMILY_MODELS.items():
        if model_name in model_names:
            return family_name
    if model_name in M4M8_HELDOUT_EXPECTED_FAMILY:
        return M4M8_HELDOUT_EXPECTED_FAMILY[model_name]
    return ""


def _pooled_split_indices(experiment, model_names, split_name: str) -> np.ndarray:
    parts = []
    for model_name in model_names:
        idx = np.asarray(
            core.indices_for_data_split(experiment, model_name, split_name),
            dtype=np.int64,
        )
        if len(idx):
            parts.append(idx)
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _family_min_raw_scores(query_z, model_supports, model_names):
    model_names = tuple(model_names)
    if not model_names:
        raise ValueError("A family must contain at least one model support.")

    per_model = {
        model_name: core.raw_scores(query_z, model_supports[model_name])
        for model_name in model_names
    }
    output = {}
    for score_name in core.SCORE_KINDS:
        matrix = np.column_stack(
            [
                np.asarray(per_model[m][score_name], dtype=np.float64)
                for m in model_names
            ]
        )
        finite_matrix = np.where(np.isfinite(matrix), matrix, np.inf)
        best_col = np.argmin(finite_matrix, axis=1)
        row_idx = np.arange(len(finite_matrix))
        best_value = finite_matrix[row_idx, best_col].astype(np.float64)
        invalid = ~np.isfinite(best_value)
        best_value[invalid] = np.nan
        best_model = np.asarray(model_names, dtype=object)[best_col]
        best_model = best_model.astype(object)
        best_model[invalid] = ""
        output[score_name] = {
            "raw_score": best_value,
            "best_model": best_model,
        }
    return output


def build_m4m8_training_family_supports(latent, experiment):
    all_models = tuple(
        model_name
        for family_models in M4M8_TRAINING_FAMILY_MODELS.values()
        for model_name in family_models
    )
    model_supports, model_diag = core.build_supports(
        latent, experiment, all_models
    )

    family_supports = {}
    family_diag_rows = []
    for family_name, model_names in M4M8_TRAINING_FAMILY_MODELS.items():
        cal_idx = _pooled_split_indices(experiment, model_names, "validation")
        if len(cal_idx) < 1:
            raise ValueError(
                f"No validation subjects available to calibrate family {family_name}."
            )
        z_cal = np.asarray(latent[cal_idx], dtype=np.float64)
        cal_raw = _family_min_raw_scores(
            z_cal, model_supports, model_names
        )
        calibration_scores = {
            score_name: np.asarray(values["raw_score"], dtype=np.float64)
            for score_name, values in cal_raw.items()
        }
        family_supports[family_name] = {
            "family_label": M4M8_TRAINING_FAMILY_LABELS[family_name],
            "model_names": tuple(model_names),
            "model_supports": model_supports,
            "calibration_scores": calibration_scores,
        }
        family_diag_rows.append(
            {
                "candidate_family": family_name,
                "candidate_family_label": M4M8_TRAINING_FAMILY_LABELS[
                    family_name
                ],
                "models": ",".join(model_names),
                "n_models": len(model_names),
                "n_calibration": len(cal_idx),
            }
        )

    model_diag = model_diag.copy()
    if not model_diag.empty:
        model_diag.insert(0, "diagnostic_level", "member_model")
        model_diag["coarse_family"] = model_diag["model_name"].map(
            m4m8_coarse_family_for_model
        )
    family_diag = pd.DataFrame(family_diag_rows)
    if not family_diag.empty:
        family_diag.insert(0, "diagnostic_level", "family_union")
    diagnostics = pd.concat(
        [model_diag, family_diag], ignore_index=True, sort=False
    )
    return family_supports, diagnostics


def score_query_set_m4m8_training_families(
    latent,
    experiment,
    family_supports,
    query_data_name,
    query_indices,
    representation_label_value,
    protocol,
):
    query_indices = np.asarray(query_indices, dtype=np.int64)
    if len(query_indices) == 0:
        return pd.DataFrame()

    z_query = np.asarray(latent[query_indices], dtype=np.float64)
    src = core.source_indices(experiment, query_indices)
    ids_all = core.load_subject_ids(core.TRIAL_DATA_NAME, query_data_name)
    subject_ids = ids_all[src]
    true_family = m4m8_coarse_family_for_model(query_data_name)

    rows = []
    for family_name, family_support in family_supports.items():
        family_raw = _family_min_raw_scores(
            z_query,
            family_support["model_supports"],
            family_support["model_names"],
        )
        for score_name, score_info in family_raw.items():
            raw_values = np.asarray(
                score_info["raw_score"], dtype=np.float64
            )
            best_models = np.asarray(
                score_info["best_model"], dtype=object
            )
            compatibility = core.compatibility_from_calibration(
                family_support["calibration_scores"][score_name],
                raw_values,
            )
            for local_index in range(len(query_indices)):
                rows.append(
                    {
                        "protocol": protocol,
                        "representation": representation_label_value,
                        "query_data_name": query_data_name,
                        "query_true_model": (
                            query_data_name
                            if query_data_name.startswith("m")
                            else ""
                        ),
                        "query_true_family": true_family,
                        "combined_subject_index": int(
                            query_indices[local_index]
                        ),
                        "source_subject_index": int(src[local_index]),
                        "subject_id": str(subject_ids[local_index]),
                        "candidate_family": family_name,
                        "candidate_family_label": family_support[
                            "family_label"
                        ],
                        "candidate_family_models": ",".join(
                            family_support["model_names"]
                        ),
                        "best_model_within_family": str(
                            best_models[local_index]
                        ),
                        "score": score_name,
                        "raw_score": float(raw_values[local_index]),
                        "compatibility": float(compatibility[local_index]),
                    }
                )
    return pd.DataFrame(rows)


def best_m4m8_training_family_from_long(score_table: pd.DataFrame):
    if score_table is None or score_table.empty:
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
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "best_family": best_row["candidate_family"],
                "best_family_label": best_row[
                    "candidate_family_label"
                ],
                "best_model_within_family": best_row[
                    "best_model_within_family"
                ],
                "c_max": float(best_row["compatibility"]),
            }
        )


        for family_name in M4M8_TRAINING_FAMILY_MODELS:
            fam = finite[
                finite["candidate_family"] == family_name
            ]
            if fam.empty:
                row[f"compatibility_{family_name}"] = np.nan
                row[f"raw_score_{family_name}"] = np.nan
                row[f"best_model_{family_name}"] = ""
            else:
                fam_row = fam.iloc[0]
                row[f"compatibility_{family_name}"] = float(
                    fam_row["compatibility"]
                )
                row[f"raw_score_{family_name}"] = float(
                    fam_row["raw_score"]
                )
                row[f"best_model_{family_name}"] = str(
                    fam_row["best_model_within_family"]
                )

        if (
            row.get("query_true_family")
            in M4M8_TRAINING_FAMILY_MODELS
        ):
            row["correct_family"] = bool(
                row["best_family"] == row["query_true_family"]
            )
        else:
            row["correct_family"] = np.nan

        for threshold in core.COMPATIBILITY_THRESHOLDS:
            row[f"outside_all_at_{threshold:g}"] = bool(
                row["c_max"] < threshold
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_m4m8_training_family_best(best: pd.DataFrame):
    if best is None or best.empty:
        return pd.DataFrame()
    rows = []
    for (rep, score_name, query_name), group in best.groupby(
        ["representation", "score", "query_data_name"], sort=False
    ):
        row = {
            "query_data_name": query_name,
            "query_true_family": str(
                group["query_true_family"].iloc[0]
            ),
            "representation": rep,
            "score": score_name,
            "n_subjects": len(group),
            "mean_c_max": float(group["c_max"].mean()),
            "median_c_max": float(group["c_max"].median()),
        }
        for threshold in core.COMPATIBILITY_THRESHOLDS:
            row[f"prop_outside_all_at_{threshold:g}"] = float(
                (group["c_max"] < threshold).mean()
            )
        for family_name in M4M8_TRAINING_FAMILY_MODELS:
            row[f"prop_best_{family_name}"] = float(
                (group["best_family"] == family_name).mean()
            )
            compat_col = f"compatibility_{family_name}"
            raw_col = f"raw_score_{family_name}"
            if compat_col in group:
                row[f"mean_compatibility_{family_name}"] = float(
                    group[compat_col].mean()
                )
                row[f"median_compatibility_{family_name}"] = float(
                    group[compat_col].median()
                )
            if raw_col in group:
                row[f"mean_raw_score_{family_name}"] = float(
                    group[raw_col].mean()
                )
                row[f"median_raw_score_{family_name}"] = float(
                    group[raw_col].median()
                )

        known = group["correct_family"].dropna()
        row["family_accuracy"] = (
            float(known.astype(float).mean())
            if len(known)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_m4m8_training_family_distances(scores: pd.DataFrame):
    if scores is None or scores.empty:
        return pd.DataFrame()
    rows = []
    for (
        rep,
        score_name,
        query_name,
        family_name,
    ), group in scores.groupby(
        [
            "representation",
            "score",
            "query_data_name",
            "candidate_family",
        ],
        sort=False,
    ):
        rows.append(
            {
                "query_data_name": query_name,
                "representation": rep,
                "score": score_name,
                "candidate_family": family_name,
                "candidate_family_label": M4M8_TRAINING_FAMILY_LABELS[
                    family_name
                ],
                "candidate_family_models": ",".join(
                    M4M8_TRAINING_FAMILY_MODELS[family_name]
                ),
                "n_subjects": int(
                    group["combined_subject_index"].nunique()
                ),
                "mean_raw_score": float(group["raw_score"].mean()),
                "median_raw_score": float(
                    group["raw_score"].median()
                ),
                "mean_compatibility": float(
                    group["compatibility"].mean()
                ),
                "median_compatibility": float(
                    group["compatibility"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_m4m8_known_family_recovery(best: pd.DataFrame):
    if best is None or best.empty:
        return pd.DataFrame()
    known = best[
        best["query_data_name"].isin(M4M8_HELDOUT_EXPECTED_FAMILY)
    ].copy()
    if known.empty:
        return pd.DataFrame()
    rows = []
    for (rep, score_name), group in known.groupby(
        ["representation", "score"], sort=False
    ):
        rows.append(
            {
                "representation": rep,
                "score": score_name,
                "n_subjects": len(group),
                "family_accuracy": float(
                    group["correct_family"].astype(float).mean()
                ),
                **{
                    f"accuracy_{model_name}": float(
                        group.loc[
                            group["query_data_name"] == model_name,
                            "correct_family",
                        ]
                        .astype(float)
                        .mean()
                    )
                    for model_name in M4M8_HELDOUT_EXPECTED_FAMILY
                },
            }
        )
    return pd.DataFrame(rows)


def m4m8_family_query_indices(experiment, query_name: str):
    if query_name in M4M8_HELDOUT_EXPECTED_FAMILY:
        return core.indices_for_data_split(
            experiment, query_name, "test"
        )
    return application_query_indices(experiment, query_name)


def run_m4m8_training_family_analysis(experiment, rep_outputs):
    query_names = tuple(M4M8_HELDOUT_EXPECTED_FAMILY) + tuple(
        application_query_names()
    )
    training_models = tuple(
        model_name
        for family_models in M4M8_TRAINING_FAMILY_MODELS.values()
        for model_name in family_models
    )

    query_score_tables = []
    sim_score_tables = []
    diagnostics = []

    for rep_label, info in rep_outputs.items():
        family_supports, diag = build_m4m8_training_family_supports(
            info["latent"], experiment
        )
        diag.insert(0, "representation", rep_label)
        diagnostics.append(diag)

        for query_name in query_names:
            query_idx = m4m8_family_query_indices(
                experiment, query_name
            )
            query_score_tables.append(
                score_query_set_m4m8_training_families(
                    info["latent"],
                    experiment,
                    family_supports,
                    query_name,
                    query_idx,
                    rep_label,
                    "m4m8_training_family_query",
                )
            )


        for model_name in training_models:
            query_idx = core.indices_for_data_split(
                experiment, model_name, "test"
            )
            sim_score_tables.append(
                score_query_set_m4m8_training_families(
                    info["latent"],
                    experiment,
                    family_supports,
                    model_name,
                    query_idx,
                    rep_label,
                    "m4m8_training_family_simulation_baseline",
                )
            )

    query_scores = (
        pd.concat(query_score_tables, ignore_index=True)
        if query_score_tables
        else pd.DataFrame()
    )
    sim_scores = (
        pd.concat(sim_score_tables, ignore_index=True)
        if sim_score_tables
        else pd.DataFrame()
    )
    query_best = best_m4m8_training_family_from_long(query_scores)
    sim_best = best_m4m8_training_family_from_long(sim_scores)

    query_summary = summarize_m4m8_training_family_best(query_best)
    distance_summary = summarize_m4m8_training_family_distances(
        query_scores
    )
    known_recovery_summary = summarize_m4m8_known_family_recovery(
        query_best
    )
    sim_summary = summarize_m4m8_training_family_best(sim_best)

    application_best = query_best[
        query_best["query_data_name"].isin(application_query_names())
    ].copy()
    application_scores = query_scores[
        query_scores["query_data_name"].isin(application_query_names())
    ].copy()
    heldout_best = query_best[
        query_best["query_data_name"].isin(
            M4M8_HELDOUT_EXPECTED_FAMILY
        )
    ].copy()
    heldout_scores = query_scores[
        query_scores["query_data_name"].isin(
            M4M8_HELDOUT_EXPECTED_FAMILY
        )
    ].copy()

    diagnostics_frame = (
        pd.concat(diagnostics, ignore_index=True)
        if diagnostics
        else pd.DataFrame()
    )
    return {
        "query_scores": query_scores,
        "query_best": query_best,
        "query_summary": query_summary,
        "distance_summary": distance_summary,
        "known_recovery_summary": known_recovery_summary,
        "heldout_scores": heldout_scores,
        "heldout_best": heldout_best,
        "application_scores": application_scores,
        "application_best": application_best,
        "simulation_scores": sim_scores,
        "simulation_best": sim_best,
        "simulation_summary": sim_summary,
        "diagnostics": diagnostics_frame,
    }


def candidate_ood_group_map() -> dict[str, str]:
    return {
        **{m: "id" for m in CANDIDATE_OOD_ID},
        **{m: "related_unseen" for m in CANDIDATE_OOD_RELATED_UNSEEN},
        **{m: "novel_family" for m in CANDIDATE_OOD_NOVEL_FAMILY},
    }


def run_candidate_set_ood(experiment, recovery_rep_outputs):
    group_map = candidate_ood_group_map()
    score_tables = []
    uncertainty_rows = []

    for rep_label, info in recovery_rep_outputs.items():
        supports, _ = core.build_supports(
            info["latent"], experiment, CANDIDATE_OOD_CANDIDATES
        )
        for query_name, group_label in group_map.items():
            query_idx = core.indices_for_data_split(
                experiment, query_name, "test"
            )
            table = core.score_query_set(
                info["latent"],
                experiment,
                supports,
                query_name,
                query_idx,
                rep_label,
                "candidate_set_ood",
            )
            table["ood_group"] = group_label
            score_tables.append(table)


            probs = info.get("probabilities")
            rep = info.get("rep")
            model_names = (
                tuple(rep.classifier_model_names)
                if rep is not None and rep.classifier_model_names is not None
                else None
            )
            if probs is None or model_names is None:
                continue
            if not set(CANDIDATE_OOD_CANDIDATES) <= set(model_names):
                continue
            cols = [model_names.index(m) for m in CANDIDATE_OOD_CANDIDATES]
            p = np.asarray(probs[query_idx][:, cols], dtype=np.float64)


            denom = np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
            p = p / denom
            max_prob = p.max(axis=1)
            entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
            if p.shape[1] > 1:
                entropy = entropy / math.log(p.shape[1])
            src = core.source_indices(experiment, query_idx)
            for i in range(len(query_idx)):
                uncertainty_rows.append(
                    {
                        "representation": rep_label,
                        "query_data_name": query_name,
                        "ood_group": group_label,
                        "combined_subject_index": int(query_idx[i]),
                        "source_subject_index": int(src[i]),
                        "max_probability": float(max_prob[i]),
                        "one_minus_max_probability": float(1.0 - max_prob[i]),
                        "normalized_entropy": float(entropy[i]),
                    }
                )

    scores = (
        pd.concat(score_tables, ignore_index=True)
        if score_tables
        else pd.DataFrame()
    )
    best = core.best_from_long(scores)
    if not best.empty:
        best["ood_group"] = best["query_data_name"].map(group_map)
    uncertainty = pd.DataFrame(uncertainty_rows)
    latent_summary = summarize_candidate_ood_latent(best)
    uncertainty_summary = summarize_classifier_uncertainty(uncertainty)
    return scores, best, latent_summary, uncertainty, uncertainty_summary


def _auroc_between_groups(frame, negative_group, positive_group, score_col):
    neg = frame[frame["ood_group"] == negative_group]
    pos = frame[frame["ood_group"] == positive_group]
    if neg.empty or pos.empty:
        return np.nan, 0
    y = np.r_[np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)]
    score = np.r_[neg[score_col].to_numpy(float), pos[score_col].to_numpy(float)]
    return core.binary_auroc(y, score), len(y)


def summarize_candidate_ood_latent(best):
    if best.empty:
        return pd.DataFrame()
    rows = []
    for (rep, score_name), group in best.groupby(
        ["representation", "score"], sort=False
    ):
        for group_name, subset in group.groupby("ood_group", sort=False):
            rows.append(
                {
                    "representation": rep,
                    "method": f"latent::{score_name}",
                    "statistic": f"group::{group_name}",
                    "n": len(subset),
                    "value": float(subset["c_max"].mean()),
                    "median": float(subset["c_max"].median()),
                }
            )
        temp = group.copy()
        temp["ood_score"] = 1.0 - temp["c_max"]
        for neg, pos in (
            ("id", "related_unseen"),
            ("id", "novel_family"),
            ("related_unseen", "novel_family"),
        ):
            auc, n = _auroc_between_groups(temp, neg, pos, "ood_score")
            rows.append(
                {
                    "representation": rep,
                    "method": f"latent::{score_name}",
                    "statistic": f"AUROC::{neg}_vs_{pos}",
                    "n": n,
                    "value": auc,
                    "median": np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_classifier_uncertainty(frame):
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for rep, group in frame.groupby("representation", sort=False):
        for metric in ("one_minus_max_probability", "normalized_entropy"):
            for neg, pos in (
                ("id", "related_unseen"),
                ("id", "novel_family"),
                ("related_unseen", "novel_family"),
            ):
                auc, n = _auroc_between_groups(group, neg, pos, metric)
                rows.append(
                    {
                        "representation": rep,
                        "method": f"classifier_uncertainty::{metric}",
                        "statistic": f"AUROC::{neg}_vs_{pos}",
                        "n": n,
                        "value": auc,
                    }
                )
    return pd.DataFrame(rows)


def classifier_head_recovery(experiment, recovery_rep_outputs):
    rows = []
    subject_rows = []
    for rep_label, info in recovery_rep_outputs.items():
        rep = info.get("rep")
        probs = info.get("probabilities")
        if rep is None or probs is None or rep.classifier_model_names is None:
            continue
        model_names = tuple(rep.classifier_model_names)
        if not set(CANDIDATE_OOD_CANDIDATES) <= set(model_names):
            continue
        cols = [model_names.index(m) for m in CANDIDATE_OOD_CANDIDATES]
        all_true = []
        all_pred = []
        for query_name in CANDIDATE_OOD_ID:
            idx = core.indices_for_data_split(experiment, query_name, "test")
            p = np.asarray(probs[idx][:, cols], dtype=np.float64)
            p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
            predicted = np.asarray(CANDIDATE_OOD_CANDIDATES)[np.argmax(p, axis=1)]
            src = core.source_indices(experiment, idx)
            for i, pred in enumerate(predicted):
                subject_rows.append(
                    {
                        "representation": rep_label,
                        "query_true_model": query_name,
                        "source_subject_index": int(src[i]),
                        "predicted_model": str(pred),
                        "correct_exact": bool(pred == query_name),
                        "correct_family": bool(
                            core.model_family(str(pred)) == core.model_family(query_name)
                        ),
                    }
                )
            all_true.extend([query_name] * len(predicted))
            all_pred.extend(predicted.tolist())
        if all_true:
            exact = np.mean(np.asarray(all_true) == np.asarray(all_pred))
            fam = np.mean(
                [
                    core.model_family(p) == core.model_family(t)
                    for t, p in zip(all_true, all_pred)
                ]
            )
            rows.append(
                {
                    "representation": rep_label,
                    "n_subjects": len(all_true),
                    "exact_model_accuracy": float(exact),
                    "family_accuracy": float(fam),
                }
            )
    return pd.DataFrame(subject_rows), pd.DataFrame(rows)


def real_vs_simulation_subject_table(rec_best, real_best):
    if rec_best is None or real_best is None or rec_best.empty or real_best.empty:
        return pd.DataFrame()

    required = {"representation", "score", "c_max"}
    missing_rec = required - set(rec_best.columns)
    missing_real = required - set(real_best.columns)
    if missing_rec or missing_real:
        warnings.warn(
            "Cannot compute real-vs-simulation AUROC because required columns "
            f"are missing. recovery missing={sorted(missing_rec)}, "
            f"real missing={sorted(missing_real)}"
        )
        return pd.DataFrame()

    sim = rec_best.copy()
    sim["data_origin"] = "simulation"
    sim["real_vs_sim_label"] = 0

    real = real_best.copy()
    real["data_origin"] = "real"
    real["real_vs_sim_label"] = 1

    frame = pd.concat([sim, real], ignore_index=True, sort=False)
    frame["c_max"] = pd.to_numeric(frame["c_max"], errors="coerce")
    frame = frame[np.isfinite(frame["c_max"].to_numpy(float))].copy()
    frame["real_vs_sim_score"] = 1.0 - frame["c_max"].to_numpy(float)
    return frame


def summarize_real_vs_simulation(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()

    app_names = [
        name
        for name in application_query_names()
        if name
        in set(
            frame.loc[frame["data_origin"] == "real", "query_data_name"]
            .dropna()
            .astype(str)
        )
    ]
    if not app_names:
        app_names = [None]

    rows = []
    for app_name in app_names:
        for (rep, score_name), group in frame.groupby(
            ["representation", "score"], dropna=False, sort=False
        ):
            sim = group[group["data_origin"] == "simulation"]
            real = group[group["data_origin"] == "real"]
            if app_name is not None:
                real = real[real["query_data_name"].astype(str) == str(app_name)]
            if sim.empty or real.empty:
                continue

            y = np.r_[
                np.zeros(len(sim), dtype=int),
                np.ones(len(real), dtype=int),
            ]
            scores = np.r_[
                sim["real_vs_sim_score"].to_numpy(float),
                real["real_vs_sim_score"].to_numpy(float),
            ]
            auc = core.binary_auroc(y, scores)

            rows.append(
                {
                    "application_query": (
                        app_name if app_name is not None else "all"
                    ),
                    "representation": rep,
                    "score": score_name,
                    "n_simulation": int(len(sim)),
                    "n_real": int(len(real)),
                    "mean_c_max_simulation": float(sim["c_max"].mean()),
                    "median_c_max_simulation": float(sim["c_max"].median()),
                    "mean_c_max_real": float(real["c_max"].mean()),
                    "median_c_max_real": float(real["c_max"].median()),
                    "median_c_max_gap_real_minus_simulation": float(
                        real["c_max"].median() - sim["c_max"].median()
                    ),
                    "auroc_real_vs_simulation": float(auc),
                }
            )
    return pd.DataFrame(rows)


def run_real_vs_simulation_analysis(rec_best, real_best):
    subjects = real_vs_simulation_subject_table(rec_best, real_best)
    summary = summarize_real_vs_simulation(subjects)
    return subjects, summary


def percentile_interval(values, level=CI_LEVEL):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    alpha = (1.0 - level) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def bootstrap_binary_mean(frame, success_col, group_cols, label):
    if frame.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        values = group[success_col].astype(float).to_numpy()
        if len(values) == 0:
            continue
        boots = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
        for b in range(BOOTSTRAP_SAMPLES):
            sample = rng.integers(0, len(values), len(values))
            boots[b] = values[sample].mean()
        lo, hi = percentile_interval(boots)
        row = {name: value for name, value in zip(group_cols, key)}
        row.update(
            {
                "statistic": label,
                "estimate": float(values.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "n": len(values),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_auroc(frame, group_cols, group_col, negative, positive, score_col, label):
    if frame.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        neg = group[group[group_col] == negative]
        pos = group[group[group_col] == positive]
        if neg.empty or pos.empty:
            continue
        neg_scores = neg[score_col].to_numpy(float)
        pos_scores = pos[score_col].to_numpy(float)
        y = np.r_[np.zeros(len(neg_scores), int), np.ones(len(pos_scores), int)]
        scores = np.r_[neg_scores, pos_scores]
        estimate = core.binary_auroc(y, scores)
        boots = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
        for b in range(BOOTSTRAP_SAMPLES):
            ni = rng.integers(0, len(neg_scores), len(neg_scores))
            pi = rng.integers(0, len(pos_scores), len(pos_scores))
            yy = np.r_[np.zeros(len(ni), int), np.ones(len(pi), int)]
            ss = np.r_[neg_scores[ni], pos_scores[pi]]
            boots[b] = core.binary_auroc(yy, ss)
        lo, hi = percentile_interval(boots)
        row = {name: value for name, value in zip(group_cols, key)}
        row.update(
            {
                "statistic": label,
                "estimate": float(estimate),
                "ci_low": lo,
                "ci_high": hi,
                "n": len(y),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_real_vs_simulation(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    parts = []
    sim = frame[frame["data_origin"] == "simulation"]
    for app_name in application_query_names():
        real = frame[
            (frame["data_origin"] == "real")
            & (frame["query_data_name"].astype(str) == str(app_name))
        ]
        if sim.empty or real.empty:
            continue
        subset = pd.concat([sim, real], ignore_index=True)
        result = bootstrap_auroc(
            subset,
            ["representation", "score"],
            "data_origin",
            "simulation",
            "real",
            "real_vs_sim_score",
            "real_vs_simulation_AUROC",
        )
        if not result.empty:
            result.insert(0, "application_query", app_name)
            parts.append(add_metadata(result, analysis="real_vs_simulation"))
    return (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    )

def bootstrap_primary_results(gen_best, rec_best, mis_best, real_best, candidate_ood_best):
    parts = []

    if not gen_best.empty:
        temp = gen_best.copy()
        temp["success"] = temp["best_family"] == temp["query_true_family"]
        parts.append(
            add_metadata(
                bootstrap_binary_mean(
                    temp,
                    "success",
                    ["representation", "score"],
                    "generalization_family_accuracy",
                ),
                analysis="generalization",
            )
        )

    if not rec_best.empty:
        temp = rec_best.copy()
        temp["exact_success"] = temp["best_model"] == temp["query_true_model"]
        temp["family_success"] = temp["best_family"] == temp["query_true_family"]
        parts.append(
            add_metadata(
                bootstrap_binary_mean(
                    temp,
                    "exact_success",
                    ["representation", "score"],
                    "recovery_exact_accuracy",
                ),
                analysis="recovery",
            )
        )
        parts.append(
            add_metadata(
                bootstrap_binary_mean(
                    temp,
                    "family_success",
                    ["representation", "score"],
                    "recovery_family_accuracy",
                ),
                analysis="recovery",
            )
        )

    if not mis_best.empty:
        temp = mis_best.copy()
        temp["ood_score"] = 1.0 - temp["c_max"]
        parts.append(
            add_metadata(
                bootstrap_auroc(
                    temp,
                    ["representation", "score"],
                    "coverage_group",
                    "id",
                    "different_family_unseen",
                    "ood_score",
                    "misspec_AUROC_id_vs_different_family",
                ),
                analysis="family_coverage",
            )
        )
        parts.append(
            add_metadata(
                bootstrap_auroc(
                    temp,
                    ["representation", "score"],
                    "coverage_group",
                    "same_family_unseen",
                    "different_family_unseen",
                    "ood_score",
                    "misspec_AUROC_same_family_vs_different_family",
                ),
                analysis="family_coverage",
            )
        )

    if not candidate_ood_best.empty:
        temp = candidate_ood_best.copy()
        temp["ood_score"] = 1.0 - temp["c_max"]
        parts.append(
            add_metadata(
                bootstrap_auroc(
                    temp,
                    ["representation", "score"],
                    "ood_group",
                    "id",
                    "novel_family",
                    "ood_score",
                    "candidate_OOD_AUROC_id_vs_novel_family",
                ),
                analysis="candidate_set_ood",
            )
        )

    if not real_best.empty:
        temp = real_best.copy()
        temp["outside_005"] = temp["c_max"] < 0.05
        group_cols = ["representation", "score"]
        if "query_data_name" in temp.columns:
            group_cols = ["query_data_name", "representation", "score"]
        parts.append(
            add_metadata(
                bootstrap_binary_mean(
                    temp,
                    "outside_005",
                    group_cols,
                    "application_prop_outside_all_at_0.05",
                ),
                analysis="application",
            )
        )

    parts = [part for part in parts if part is not None and not part.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def make_experiment(trial_data_name: str, split_seed: int):
    global ACTIVE_APPLICATION_QUERIES
    configure_core(trial_data_name, split_seed, PRIMARY_K)
    ACTIVE_APPLICATION_QUERIES = available_application_query_names(
        trial_data_name
    )
    all_data_names = application_data_names_to_load()
    log(f"  Active application queries: {ACTIVE_APPLICATION_QUERIES}")
    return core.load_experiment_data(
        trial_data_name,
        all_data_names,
        seed=split_seed,
        full_real=True,
    )


def encode_all_for_trial(experiment):
    log("  Encoding generalization representations (train m4--m8)...")
    generalization_reps = core.encode_protocol_representations(
        experiment, GENERALIZATION_TRAIN_MODELS
    )
    log("  Encoding recovery representations (train m1--m3)...")
    recovery_reps = core.encode_protocol_representations(
        experiment, CANDIDATE_OOD_CANDIDATES
    )
    return generalization_reps, recovery_reps


def run_primary_dataset(trial_data_name: str):
    configure_core(trial_data_name, PRIMARY_SPLIT_SEED, PRIMARY_K)
    run_dir = ensure_dir(OUTPUT_ROOT / "primary" / trial_data_name)

    log(f"\n========== PRIMARY: {trial_data_name} ==========")
    cov_gen = checkpoint_coverage(GENERALIZATION_TRAIN_MODELS, "generalization")
    cov_rec = checkpoint_coverage(CANDIDATE_OOD_CANDIDATES, "recovery")
    save_frame(pd.concat([cov_gen, cov_rec]), run_dir / "checkpoint_coverage.csv")

    experiment = make_experiment(trial_data_name, PRIMARY_SPLIT_SEED)
    generalization_reps, recovery_reps = encode_all_for_trial(experiment)

    core.SCORE_KINDS = tuple(PRIMARY_SCORE_KINDS)
    core.KNN_K = PRIMARY_K

    gen_scores, gen_best, gen_summary, gen_diag = core.run_generalization(
        experiment, generalization_reps
    )

    (
        rec_scores,
        rec_best,
        rec_summary,
        rec_agreement,
        rec_bic,
        rec_clf,
        rec_diag,
    ) = core.run_recovery(experiment, recovery_reps)

    mis_scores, mis_best, mis_summary, mis_diag = core.run_misspec(
        experiment, generalization_reps
    )

    (
        real_scores,
        real_best,
        real_summary,
        real_bic,
        real_clf,
        real_diag,
    ) = run_application(experiment, recovery_reps)

    real_vs_sim_subjects, real_vs_sim_summary = (
        run_real_vs_simulation_analysis(rec_best, real_best)
    )


    log("  Parallel application analysis with encoder trained on m4--m8...")
    (
        m4m8_rec_scores,
        m4m8_rec_best,
        m4m8_rec_summary,
    ) = run_recovery_latent_only(experiment, generalization_reps)
    (
        m4m8_app_scores,
        m4m8_app_best,
        m4m8_app_summary,
    ) = run_real_latent_only(experiment, generalization_reps)
    (
        m4m8_real_vs_sim_subjects,
        m4m8_real_vs_sim_summary,
    ) = run_real_vs_simulation_analysis(m4m8_rec_best, m4m8_app_best)


    log("  M4--M8 training-family analysis (option-wise vs attribute-wise)...")
    m4m8_family = run_m4m8_training_family_analysis(
        experiment, generalization_reps
    )
    (
        m4m8_family_app_vs_sim_subjects,
        m4m8_family_app_vs_sim_summary,
    ) = run_real_vs_simulation_analysis(
        m4m8_family["simulation_best"],
        m4m8_family["application_best"],
    )

    (
        ood_scores,
        ood_best,
        ood_summary,
        uncertainty,
        uncertainty_summary,
    ) = run_candidate_set_ood(experiment, recovery_reps)

    head_subjects, head_summary = classifier_head_recovery(
        experiment, recovery_reps
    )

    bootstrap = bootstrap_primary_results(
        gen_best, rec_best, mis_best, real_best, ood_best
    )
    real_vs_sim_bootstrap = bootstrap_real_vs_simulation(real_vs_sim_subjects)
    bootstrap_parts = [
        x for x in (bootstrap, real_vs_sim_bootstrap)
        if x is not None and not x.empty
    ]
    bootstrap = (
        pd.concat(bootstrap_parts, ignore_index=True)
        if bootstrap_parts else pd.DataFrame()
    )


    empty = pd.DataFrame()
    m4m8_bootstrap = bootstrap_primary_results(
        empty,
        m4m8_rec_best,
        empty,
        m4m8_app_best,
        empty,
    )
    m4m8_real_vs_sim_bootstrap = bootstrap_real_vs_simulation(
        m4m8_real_vs_sim_subjects
    )
    m4m8_bootstrap_parts = [
        x
        for x in (m4m8_bootstrap, m4m8_real_vs_sim_bootstrap)
        if x is not None and not x.empty
    ]
    m4m8_bootstrap = (
        pd.concat(m4m8_bootstrap_parts, ignore_index=True)
        if m4m8_bootstrap_parts
        else pd.DataFrame()
    )


    m4m8_family_bootstrap = bootstrap_primary_results(
        empty,
        empty,
        empty,
        m4m8_family["application_best"],
        empty,
    )
    m4m8_family_app_vs_sim_bootstrap = bootstrap_real_vs_simulation(
        m4m8_family_app_vs_sim_subjects
    )
    m4m8_family_bootstrap_parts = [
        x
        for x in (
            m4m8_family_bootstrap,
            m4m8_family_app_vs_sim_bootstrap,
        )
        if x is not None and not x.empty
    ]
    m4m8_family_bootstrap = (
        pd.concat(m4m8_family_bootstrap_parts, ignore_index=True)
        if m4m8_family_bootstrap_parts
        else pd.DataFrame()
    )


    m4m8_rec_summary = tag_m4m8_parallel(m4m8_rec_summary)
    m4m8_app_summary = tag_m4m8_parallel(m4m8_app_summary)
    m4m8_real_vs_sim_summary = tag_m4m8_parallel(
        m4m8_real_vs_sim_summary
    )
    m4m8_bootstrap = tag_m4m8_parallel(m4m8_bootstrap)
    m4m8_rec_best = tag_m4m8_parallel(m4m8_rec_best)
    m4m8_rec_scores = tag_m4m8_parallel(m4m8_rec_scores)
    m4m8_app_best = tag_m4m8_parallel(m4m8_app_best)
    m4m8_app_scores = tag_m4m8_parallel(m4m8_app_scores)
    m4m8_real_vs_sim_subjects = tag_m4m8_parallel(
        m4m8_real_vs_sim_subjects
    )


    for _key in (
        "query_scores",
        "query_best",
        "query_summary",
        "distance_summary",
        "known_recovery_summary",
        "heldout_scores",
        "heldout_best",
        "application_scores",
        "application_best",
        "simulation_scores",
        "simulation_best",
        "simulation_summary",
        "diagnostics",
    ):
        m4m8_family[_key] = tag_m4m8_training_family(
            m4m8_family[_key]
        )
    m4m8_family_app_vs_sim_subjects = tag_m4m8_training_family(
        m4m8_family_app_vs_sim_subjects
    )
    m4m8_family_app_vs_sim_summary = tag_m4m8_training_family(
        m4m8_family_app_vs_sim_summary
    )
    m4m8_family_bootstrap = tag_m4m8_training_family(
        m4m8_family_bootstrap
    )


    if not real_vs_sim_summary.empty:
        primary_rows = real_vs_sim_summary[
            real_vs_sim_summary["score"] == "knn"
        ]
        for _, row in primary_rows.iterrows():
            if real_vs_sim_bootstrap.empty:
                ci_match = pd.DataFrame()
            elif "application_query" in real_vs_sim_bootstrap.columns:
                ci_match = real_vs_sim_bootstrap[
                    (real_vs_sim_bootstrap["representation"] == row["representation"])
                    & (real_vs_sim_bootstrap["score"] == row["score"])
                    & (
                        real_vs_sim_bootstrap["application_query"]
                        == row.get("application_query")
                    )
                ]
            else:
                ci_match = real_vs_sim_bootstrap[
                    (real_vs_sim_bootstrap["representation"] == row["representation"])
                    & (real_vs_sim_bootstrap["score"] == row["score"])
                ]
            if not ci_match.empty:
                ci_row = ci_match.iloc[0]
                ci_text = (
                    f" [{ci_row['ci_low']:.3f}, {ci_row['ci_high']:.3f}]"
                )
            else:
                ci_text = ""
            log(
                "  Application-vs-simulation kNN "
                f"[{row.get('application_query', 'all')}]: "
                f"{row['representation']} AUROC="
                f"{row['auroc_real_vs_simulation']:.3f}{ci_text}; "
                f"median Cmax sim={row['median_c_max_simulation']:.3f}, "
                f"app={row['median_c_max_real']:.3f}"
            )


    if not m4m8_real_vs_sim_summary.empty:
        primary_rows = m4m8_real_vs_sim_summary[
            m4m8_real_vs_sim_summary["score"] == "knn"
        ]
        for _, row in primary_rows.iterrows():
            if m4m8_real_vs_sim_bootstrap.empty:
                ci_match = pd.DataFrame()
            elif "application_query" in m4m8_real_vs_sim_bootstrap.columns:
                ci_match = m4m8_real_vs_sim_bootstrap[
                    (
                        m4m8_real_vs_sim_bootstrap["representation"]
                        == row["representation"]
                    )
                    & (m4m8_real_vs_sim_bootstrap["score"] == row["score"])
                    & (
                        m4m8_real_vs_sim_bootstrap["application_query"]
                        == row.get("application_query")
                    )
                ]
            else:
                ci_match = m4m8_real_vs_sim_bootstrap[
                    (
                        m4m8_real_vs_sim_bootstrap["representation"]
                        == row["representation"]
                    )
                    & (m4m8_real_vs_sim_bootstrap["score"] == row["score"])
                ]
            if not ci_match.empty:
                ci_row = ci_match.iloc[0]
                ci_text = (
                    f" [{ci_row['ci_low']:.3f}, {ci_row['ci_high']:.3f}]"
                )
            else:
                ci_text = ""
            log(
                "  M4--M8 encoder | application-vs-simulation kNN "
                f"[{row.get('application_query', 'all')}]: "
                f"{row['representation']} AUROC="
                f"{row['auroc_real_vs_simulation']:.3f}{ci_text}; "
                f"median Cmax sim={row['median_c_max_simulation']:.3f}, "
                f"app={row['median_c_max_real']:.3f}"
            )


    if not m4m8_family_app_vs_sim_summary.empty:
        primary_rows = m4m8_family_app_vs_sim_summary[
            m4m8_family_app_vs_sim_summary["score"] == "knn"
        ]
        for _, row in primary_rows.iterrows():
            log(
                "  M4--M8 training families | application-vs-training-sim kNN "
                f"[{row.get('application_query', 'all')}]: "
                f"{row['representation']} AUROC="
                f"{row['auroc_real_vs_simulation']:.3f}; "
                f"median Cmax train-sim="
                f"{row['median_c_max_simulation']:.3f}, "
                f"app={row['median_c_max_real']:.3f}"
            )

    if not m4m8_family["known_recovery_summary"].empty:
        primary_rows = m4m8_family["known_recovery_summary"]
        primary_rows = primary_rows[primary_rows["score"] == "knn"]
        for _, row in primary_rows.iterrows():
            log(
                "  M4--M8 training families | held-out known-family kNN: "
                f"{row['representation']} family accuracy="
                f"{row['family_accuracy']:.3f}"
            )


    summary_tables = {
        "generalization_summary.csv": gen_summary,
        "recovery_summary.csv": rec_summary,
        "recovery_agreement.csv": rec_agreement,
        "family_coverage_summary.csv": mis_summary,
        "real_summary.csv": real_summary,
        "application_summary.csv": real_summary,
        "real_vs_simulation_summary.csv": real_vs_sim_summary,
        "candidate_ood_summary.csv": ood_summary,
        "classifier_uncertainty_summary.csv": uncertainty_summary,
        "classifier_head_recovery_summary.csv": head_summary,
        "bootstrap_95ci.csv": bootstrap,
        "support_diagnostics.csv": pd.concat(
            [gen_diag, rec_diag, mis_diag, real_diag], ignore_index=True
        )
        if any(not x.empty for x in (gen_diag, rec_diag, mis_diag, real_diag))
        else pd.DataFrame(),
    }
    for name, frame in summary_tables.items():
        save_frame(frame, run_dir / name)


    m4m8_dir = ensure_dir(run_dir / "parallel_m4m8_encoder")
    m4m8_summary_tables = {
        "heldout_candidate_recovery_summary.csv": m4m8_rec_summary,
        "application_summary.csv": m4m8_app_summary,
        "application_vs_simulation_summary.csv": m4m8_real_vs_sim_summary,
        "bootstrap_95ci.csv": m4m8_bootstrap,
    }
    for name, frame in m4m8_summary_tables.items():
        save_frame(frame, m4m8_dir / name)


    m4m8_family_dir = ensure_dir(
        m4m8_dir / "training_family_support"
    )
    m4m8_family_summary_tables = {
        "heldout_known_family_recovery_summary.csv": (
            m4m8_family["known_recovery_summary"]
        ),
        "all_queries_summary.csv": m4m8_family["query_summary"],
        "all_queries_family_distance_summary.csv": (
            m4m8_family["distance_summary"]
        ),
        "application_summary.csv": (
            m4m8_family["query_summary"][
                m4m8_family["query_summary"]["query_data_name"].isin(
                    application_query_names()
                )
            ].copy()
            if not m4m8_family["query_summary"].empty
            else pd.DataFrame()
        ),
        "training_simulation_summary.csv": (
            m4m8_family["simulation_summary"]
        ),
        "application_vs_training_simulation_summary.csv": (
            m4m8_family_app_vs_sim_summary
        ),
        "bootstrap_95ci.csv": m4m8_family_bootstrap,
        "support_diagnostics.csv": m4m8_family["diagnostics"],
    }
    for name, frame in m4m8_family_summary_tables.items():
        save_frame(frame, m4m8_family_dir / name)

    if SAVE_LONG_TABLES_FOR_PRIMARY:
        long_tables = {
            "generalization_subjects.csv": gen_best,
            "generalization_scores_long.csv": gen_scores,
            "recovery_subjects.csv": rec_best,
            "recovery_scores_long.csv": rec_scores,
            "recovery_bic_long.csv": rec_bic,
            "recovery_classifier_long.csv": rec_clf,
            "family_coverage_subjects.csv": mis_best,
            "family_coverage_scores_long.csv": mis_scores,
            "real_subjects.csv": real_best,
            "real_scores_long.csv": real_scores,
            "real_bic_long.csv": real_bic,
            "real_classifier_long.csv": real_clf,
            "real_vs_simulation_subjects.csv": real_vs_sim_subjects,
            "candidate_ood_subjects.csv": ood_best,
            "candidate_ood_scores_long.csv": ood_scores,
            "classifier_uncertainty_subjects.csv": uncertainty,
            "classifier_head_recovery_subjects.csv": head_subjects,
        }
        for name, frame in long_tables.items():
            save_frame(frame, run_dir / name)

        m4m8_long_tables = {
            "heldout_candidate_recovery_subjects.csv": m4m8_rec_best,
            "heldout_candidate_recovery_scores_long.csv": m4m8_rec_scores,
            "application_subjects.csv": m4m8_app_best,
            "application_scores_long.csv": m4m8_app_scores,
            "application_vs_simulation_subjects.csv": (
                m4m8_real_vs_sim_subjects
            ),
        }
        for name, frame in m4m8_long_tables.items():
            save_frame(frame, m4m8_dir / name)

        m4m8_family_long_tables = {
            "heldout_known_family_subjects.csv": (
                m4m8_family["heldout_best"]
            ),
            "heldout_known_family_scores_long.csv": (
                m4m8_family["heldout_scores"]
            ),
            "application_subjects.csv": (
                m4m8_family["application_best"]
            ),
            "application_scores_long.csv": (
                m4m8_family["application_scores"]
            ),
            "training_simulation_subjects.csv": (
                m4m8_family["simulation_best"]
            ),
            "training_simulation_scores_long.csv": (
                m4m8_family["simulation_scores"]
            ),
            "application_vs_training_simulation_subjects.csv": (
                m4m8_family_app_vs_sim_subjects
            ),
        }
        for name, frame in m4m8_family_long_tables.items():
            save_frame(frame, m4m8_family_dir / name)

    config = {
        "trial_data_name": trial_data_name,
        "split_seed": PRIMARY_SPLIT_SEED,
        "representations": list(ALL_REPRESENTATIONS),
        "scores": list(PRIMARY_SCORE_KINDS),
        "primary_k": PRIMARY_K,
        "parameter_manifold_disabled": DISABLE_PARAMETER_MANIFOLD,
        "candidate_ood": {
            "candidates": list(CANDIDATE_OOD_CANDIDATES),
            "id": list(CANDIDATE_OOD_ID),
            "related_unseen": list(CANDIDATE_OOD_RELATED_UNSEEN),
            "novel_family": list(CANDIDATE_OOD_NOVEL_FAMILY),
        },
        "parallel_m4m8_encoder_application": {
            "encoder_train_models": list(M4M8_PARALLEL_TRAIN_MODELS),
            "candidate_support_models": list(
                M4M8_PARALLEL_CANDIDATE_MODELS
            ),
            "application_queries": list(application_query_names()),
            "note": (
                "m1--m3 are held out from encoder training; they are used "
                "only post hoc for candidate support calibration/testing."
            ),
        },
        "m4m8_training_family_support": {
            "encoder_train_models": list(M4M8_PARALLEL_TRAIN_MODELS),
            "families": {
                family_name: list(model_names)
                for family_name, model_names in
                M4M8_TRAINING_FAMILY_MODELS.items()
            },
            "heldout_known_family_queries": dict(
                M4M8_HELDOUT_EXPECTED_FAMILY
            ),
            "application_queries": list(application_query_names()),
            "family_score_definition": (
                "For each raw score, D_F(z)=min over member-model supports; "
                "D_F is calibrated using pooled validation simulations from "
                "that family. C_max is the maximum calibrated compatibility "
                "across the two families."
            ),
            "simulation_baseline": (
                "Held-out test subjects from m4--m8; train and validation "
                "subjects are used only for reference/calibration."
            ),
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    RUN_MANIFEST_ROWS.append(
        {
            "run_type": "primary",
            "trial_data_name": trial_data_name,
            "split_seed": PRIMARY_SPLIT_SEED,
            "knn_k": PRIMARY_K,
            "output_dir": str(run_dir),
            "n_generalization_representations": len(generalization_reps),
            "n_recovery_representations": len(recovery_reps),
            "parallel_m4m8_application": True,
            "status": "ok",
        }
    )

    return {
        "experiment": experiment,
        "generalization_reps": generalization_reps,
        "recovery_reps": recovery_reps,
        "gen_summary": gen_summary,
        "gen_best": gen_best,
        "rec_summary": rec_summary,
        "rec_best": rec_best,
        "mis_summary": mis_summary,
        "mis_best": mis_best,
        "real_summary": real_summary,
        "real_best": real_best,
        "real_vs_sim_summary": real_vs_sim_summary,
        "real_vs_sim_subjects": real_vs_sim_subjects,
        "m4m8_rec_summary": m4m8_rec_summary,
        "m4m8_rec_best": m4m8_rec_best,
        "m4m8_app_summary": m4m8_app_summary,
        "m4m8_app_best": m4m8_app_best,
        "m4m8_real_vs_sim_summary": m4m8_real_vs_sim_summary,
        "m4m8_real_vs_sim_subjects": m4m8_real_vs_sim_subjects,
        "m4m8_bootstrap": m4m8_bootstrap,
        "m4m8_family_query_summary": m4m8_family["query_summary"],
        "m4m8_family_distance_summary": m4m8_family["distance_summary"],
        "m4m8_family_known_recovery_summary": (
            m4m8_family["known_recovery_summary"]
        ),
        "m4m8_family_application_best": (
            m4m8_family["application_best"]
        ),
        "m4m8_family_simulation_summary": (
            m4m8_family["simulation_summary"]
        ),
        "m4m8_family_app_vs_sim_summary": (
            m4m8_family_app_vs_sim_summary
        ),
        "m4m8_family_app_vs_sim_subjects": (
            m4m8_family_app_vs_sim_subjects
        ),
        "m4m8_family_bootstrap": m4m8_family_bootstrap,
        "ood_summary": ood_summary,
        "ood_best": ood_best,
        "uncertainty_summary": uncertainty_summary,
        "head_summary": head_summary,
        "bootstrap": bootstrap,
    }


def run_knn_sensitivity(trial_data_name, primary_outputs):
    experiment = primary_outputs["experiment"]
    gen_reps = primary_outputs["generalization_reps"]
    rec_reps = primary_outputs["recovery_reps"]
    rows = []

    for k in KNN_K_VALUES:
        configure_core(trial_data_name, PRIMARY_SPLIT_SEED, k)
        core.SCORE_KINDS = ("knn",)
        core.KNN_K = int(k)
        log(f"  kNN robustness {trial_data_name}: k={k}")

        _, gen_best, gen_summary, _ = core.run_generalization(
            experiment, gen_reps
        )
        _, rec_best, rec_summary = run_recovery_latent_only(
            experiment, rec_reps
        )
        _, mis_best, mis_summary, _ = core.run_misspec(experiment, gen_reps)
        _, real_best, real_summary = run_real_latent_only(
            experiment, rec_reps
        )
        real_vs_sim_subjects, real_vs_sim_summary = (
            run_real_vs_simulation_analysis(rec_best, real_best)
        )


        (
            m4m8_rec_scores,
            m4m8_rec_best,
            m4m8_rec_summary,
        ) = run_recovery_latent_only(experiment, gen_reps)
        (
            m4m8_app_scores,
            m4m8_app_best,
            m4m8_app_summary,
        ) = run_real_latent_only(experiment, gen_reps)
        (
            m4m8_real_vs_sim_subjects,
            m4m8_real_vs_sim_summary,
        ) = run_real_vs_simulation_analysis(
            m4m8_rec_best, m4m8_app_best
        )

        m4m8_family = run_m4m8_training_family_analysis(
            experiment, gen_reps
        )
        (
            m4m8_family_app_vs_sim_subjects,
            m4m8_family_app_vs_sim_summary,
        ) = run_real_vs_simulation_analysis(
            m4m8_family["simulation_best"],
            m4m8_family["application_best"],
        )

        _, ood_best, ood_summary, uncertainty, uncertainty_summary = (
            run_candidate_set_ood(experiment, rec_reps)
        )

        for analysis, frame in (
            ("generalization", gen_summary),
            ("recovery", rec_summary),
            ("family_coverage", mis_summary),
            ("real", real_summary),
            ("real_vs_simulation", real_vs_sim_summary),
            ("m4m8_recovery", tag_m4m8_parallel(m4m8_rec_summary)),
            ("m4m8_application", tag_m4m8_parallel(m4m8_app_summary)),
            (
                "m4m8_real_vs_simulation",
                tag_m4m8_parallel(m4m8_real_vs_sim_summary),
            ),
            (
                "m4m8_training_family_queries",
                tag_m4m8_training_family(
                    m4m8_family["query_summary"]
                ),
            ),
            (
                "m4m8_training_family_known_recovery",
                tag_m4m8_training_family(
                    m4m8_family["known_recovery_summary"]
                ),
            ),
            (
                "m4m8_training_family_real_vs_simulation",
                tag_m4m8_training_family(
                    m4m8_family_app_vs_sim_summary
                ),
            ),
            ("candidate_set_ood", ood_summary),
            ("classifier_uncertainty", uncertainty_summary),
        ):
            if frame is None or frame.empty:
                continue
            tagged = add_metadata(
                frame,
                trial_data_name=trial_data_name,
                split_seed=PRIMARY_SPLIT_SEED,
                knn_k=k,
                analysis=analysis,
            )
            rows.append(tagged)


        kdir = ensure_dir(
            OUTPUT_ROOT / "robustness_knn" / trial_data_name / f"k{k}"
        )
        save_frame(gen_best, kdir / "generalization_subjects.csv")
        save_frame(rec_best, kdir / "recovery_subjects.csv")
        save_frame(mis_best, kdir / "family_coverage_subjects.csv")
        save_frame(real_best, kdir / "real_subjects.csv")
        save_frame(
            real_vs_sim_subjects,
            kdir / "real_vs_simulation_subjects.csv",
        )
        save_frame(
            real_vs_sim_summary,
            kdir / "real_vs_simulation_summary.csv",
        )
        save_frame(ood_best, kdir / "candidate_ood_subjects.csv")
        save_frame(uncertainty, kdir / "classifier_uncertainty_subjects.csv")

        m4m8_kdir = ensure_dir(kdir / "parallel_m4m8_encoder")
        save_frame(
            tag_m4m8_parallel(m4m8_rec_best),
            m4m8_kdir / "heldout_candidate_recovery_subjects.csv",
        )
        save_frame(
            tag_m4m8_parallel(m4m8_rec_summary),
            m4m8_kdir / "heldout_candidate_recovery_summary.csv",
        )
        save_frame(
            tag_m4m8_parallel(m4m8_app_best),
            m4m8_kdir / "application_subjects.csv",
        )
        save_frame(
            tag_m4m8_parallel(m4m8_app_summary),
            m4m8_kdir / "application_summary.csv",
        )
        save_frame(
            tag_m4m8_parallel(m4m8_real_vs_sim_subjects),
            m4m8_kdir / "application_vs_simulation_subjects.csv",
        )
        save_frame(
            tag_m4m8_parallel(m4m8_real_vs_sim_summary),
            m4m8_kdir / "application_vs_simulation_summary.csv",
        )

        m4m8_family_kdir = ensure_dir(
            m4m8_kdir / "training_family_support"
        )
        save_frame(
            tag_m4m8_training_family(
                m4m8_family["query_summary"]
            ),
            m4m8_family_kdir / "all_queries_summary.csv",
        )
        save_frame(
            tag_m4m8_training_family(
                m4m8_family["distance_summary"]
            ),
            m4m8_family_kdir / "all_queries_family_distance_summary.csv",
        )
        save_frame(
            tag_m4m8_training_family(
                m4m8_family["known_recovery_summary"]
            ),
            m4m8_family_kdir / "heldout_known_family_recovery_summary.csv",
        )
        save_frame(
            tag_m4m8_training_family(
                m4m8_family_app_vs_sim_summary
            ),
            m4m8_family_kdir / "application_vs_training_simulation_summary.csv",
        )

        RUN_MANIFEST_ROWS.append(
            {
                "run_type": "knn_robustness",
                "trial_data_name": trial_data_name,
                "split_seed": PRIMARY_SPLIT_SEED,
                "knn_k": k,
                "output_dir": str(kdir),
                "status": "ok",
            }
        )

    core.SCORE_KINDS = tuple(PRIMARY_SCORE_KINDS)
    core.KNN_K = PRIMARY_K
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_split_seed_robustness(trial_data_name, primary_outputs):
    if trial_data_name != PRIMARY_TRIAL_DATA_NAME:
        return pd.DataFrame()


    gen_reps = primary_outputs["generalization_reps"]
    rec_reps = primary_outputs["recovery_reps"]
    rows = []

    for split_seed in SPLIT_ROBUSTNESS_SEEDS:
        configure_core(trial_data_name, split_seed, PRIMARY_K)
        core.SCORE_KINDS = tuple(PRIMARY_SCORE_KINDS)
        experiment = make_experiment(trial_data_name, split_seed)
        log(f"  Split robustness {trial_data_name}: seed={split_seed}")

        _, _, gen_summary, _ = core.run_generalization(experiment, gen_reps)
        _, rec_best, rec_summary = run_recovery_latent_only(experiment, rec_reps)
        _, _, mis_summary, _ = core.run_misspec(experiment, gen_reps)
        _, real_best, real_summary = run_real_latent_only(experiment, rec_reps)
        _, real_vs_sim_summary = run_real_vs_simulation_analysis(
            rec_best, real_best
        )


        _, m4m8_rec_best, m4m8_rec_summary = run_recovery_latent_only(
            experiment, gen_reps
        )
        _, m4m8_app_best, m4m8_app_summary = run_real_latent_only(
            experiment, gen_reps
        )
        _, m4m8_real_vs_sim_summary = run_real_vs_simulation_analysis(
            m4m8_rec_best, m4m8_app_best
        )

        m4m8_family = run_m4m8_training_family_analysis(
            experiment, gen_reps
        )
        _, m4m8_family_app_vs_sim_summary = (
            run_real_vs_simulation_analysis(
                m4m8_family["simulation_best"],
                m4m8_family["application_best"],
            )
        )

        _, _, ood_summary, _, uncertainty_summary = run_candidate_set_ood(
            experiment, rec_reps
        )

        for analysis, frame in (
            ("generalization", gen_summary),
            ("recovery", rec_summary),
            ("family_coverage", mis_summary),
            ("real", real_summary),
            ("real_vs_simulation", real_vs_sim_summary),
            ("m4m8_recovery", tag_m4m8_parallel(m4m8_rec_summary)),
            ("m4m8_application", tag_m4m8_parallel(m4m8_app_summary)),
            (
                "m4m8_real_vs_simulation",
                tag_m4m8_parallel(m4m8_real_vs_sim_summary),
            ),
            (
                "m4m8_training_family_queries",
                tag_m4m8_training_family(
                    m4m8_family["query_summary"]
                ),
            ),
            (
                "m4m8_training_family_known_recovery",
                tag_m4m8_training_family(
                    m4m8_family["known_recovery_summary"]
                ),
            ),
            (
                "m4m8_training_family_real_vs_simulation",
                tag_m4m8_training_family(
                    m4m8_family_app_vs_sim_summary
                ),
            ),
            ("candidate_set_ood", ood_summary),
            ("classifier_uncertainty", uncertainty_summary),
        ):
            if frame is None or frame.empty:
                continue
            rows.append(
                add_metadata(
                    frame,
                    trial_data_name=trial_data_name,
                    split_seed=split_seed,
                    knn_k=PRIMARY_K,
                    analysis=analysis,
                )
            )

        sdir = ensure_dir(
            OUTPUT_ROOT
            / "robustness_split_seed"
            / trial_data_name
            / f"s{split_seed}"
        )
        m4m8_sdir = ensure_dir(sdir / "parallel_m4m8_encoder")
        save_frame(
            tag_m4m8_parallel(m4m8_rec_summary),
            m4m8_sdir / "heldout_candidate_recovery_summary.csv",
        )
        save_frame(
            tag_m4m8_parallel(m4m8_app_summary),
            m4m8_sdir / "application_summary.csv",
        )
        save_frame(
            tag_m4m8_parallel(m4m8_real_vs_sim_summary),
            m4m8_sdir / "application_vs_simulation_summary.csv",
        )

        m4m8_family_sdir = ensure_dir(
            m4m8_sdir / "training_family_support"
        )
        save_frame(
            tag_m4m8_training_family(
                m4m8_family["query_summary"]
            ),
            m4m8_family_sdir / "all_queries_summary.csv",
        )
        save_frame(
            tag_m4m8_training_family(
                m4m8_family["distance_summary"]
            ),
            m4m8_family_sdir / "all_queries_family_distance_summary.csv",
        )
        save_frame(
            tag_m4m8_training_family(
                m4m8_family["known_recovery_summary"]
            ),
            m4m8_family_sdir / "heldout_known_family_recovery_summary.csv",
        )
        save_frame(
            tag_m4m8_training_family(
                m4m8_family_app_vs_sim_summary
            ),
            m4m8_family_sdir / "application_vs_training_simulation_summary.csv",
        )

        RUN_MANIFEST_ROWS.append(
            {
                "run_type": "split_seed_robustness",
                "trial_data_name": trial_data_name,
                "split_seed": split_seed,
                "knn_k": PRIMARY_K,
                "output_dir": str(sdir),
                "status": "ok",
            }
        )

    core.SEED = PRIMARY_SPLIT_SEED
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def concat_tagged(primary_by_trial, key):
    frames = []
    for trial, outputs in primary_by_trial.items():
        frame = outputs.get(key)
        if frame is not None and not frame.empty:
            frames.append(add_metadata(frame, trial_data_name=trial))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_split_robustness(frame):
    if frame.empty:
        return pd.DataFrame()
    numeric_cols = [
        c
        for c in (
            "family_accuracy",
            "exact_model_accuracy",
            "mean_c_max",
            "median_c_max",
            "value",
            "auroc_real_vs_simulation",
            "mean_c_max_simulation",
            "median_c_max_simulation",
            "mean_c_max_real",
            "median_c_max_real",
            "median_c_max_gap_real_minus_simulation",
            "prop_outside_all_at_0.05",
            "prop_outside_all_at_0.01",
        )
        if c in frame.columns
    ]
    id_cols = [
        c
        for c in (
            "trial_data_name",
            "analysis",
            "encoder_protocol",
            "application_query",
            "query_data_name",
            "representation",
            "score",
            "method",
            "statistic",
        )
        if c in frame.columns
    ]
    if not numeric_cols or not id_cols:
        return pd.DataFrame()
    rows = []
    for key, group in frame.groupby(id_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {name: value for name, value in zip(id_cols, key)}
        row["n_split_seeds"] = int(group["split_seed"].nunique())
        for col in numeric_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
            row[f"{col}_min"] = float(vals.min()) if vals.notna().any() else np.nan
            row[f"{col}_max"] = float(vals.max()) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_summary_figures(primary_by_trial, split_agg, knn_frame):
    figure_dir = ensure_dir(OUTPUT_ROOT / "figures")
    figures = []


    rec = concat_tagged(primary_by_trial, "rec_summary")
    if not rec.empty:
        rec = rec[(rec.get("score", "") == "knn") | (rec["representation"] == "")].copy()
        latent = rec[rec["representation"] != ""].copy()
        if not latent.empty:
            fig, ax = plt.subplots(figsize=(11, 5.6))
            labels = [
                f"{t}\n{r}"
                for t, r in zip(latent["trial_data_name"], latent["representation"])
            ]
            x = np.arange(len(latent))
            width = 0.38
            ax.bar(x - width / 2, latent["exact_model_accuracy"], width, label="Exact model")
            ax.bar(x + width / 2, latent["family_accuracy"], width, label="Family")
            ax.set_xticks(x, labels=labels, rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Recovery accuracy")
            ax.set_title("Primary kNN latent-support recovery")
            ax.legend()
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            path = figure_dir / "final_recovery_knn.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            figures.append(fig)


    ood = concat_tagged(primary_by_trial, "ood_summary")
    unc = concat_tagged(primary_by_trial, "uncertainty_summary")
    selected = []
    if not ood.empty:
        selected.append(
            ood[
                (ood["method"] == "latent::knn")
                & (ood["statistic"] == "AUROC::id_vs_novel_family")
            ][["trial_data_name", "representation", "method", "value"]]
        )
    if not unc.empty:
        selected.append(
            unc[
                (unc["method"] == "classifier_uncertainty::normalized_entropy")
                & (unc["statistic"] == "AUROC::id_vs_novel_family")
            ][["trial_data_name", "representation", "method", "value"]]
        )
    selected = [x for x in selected if not x.empty]
    if selected:
        frame = pd.concat(selected, ignore_index=True)
        fig, ax = plt.subplots(figsize=(11, 5.4))
        labels = [
            f"{t}\n{r}\n{m.replace('classifier_uncertainty::', 'entropy').replace('latent::', '')}"
            for t, r, m in zip(
                frame["trial_data_name"], frame["representation"], frame["method"]
            )
        ]
        ax.bar(np.arange(len(frame)), frame["value"])
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(frame)), labels=labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUROC")
        ax.set_title("Novel-family detection: latent support vs classifier uncertainty")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_dir / "final_candidate_ood_auc.png", dpi=220, bbox_inches="tight")
        figures.append(fig)


    rvs = concat_tagged(primary_by_trial, "real_vs_sim_summary")
    boot = concat_tagged(primary_by_trial, "bootstrap")
    if not rvs.empty:
        frame = rvs[rvs["score"] == "knn"].copy()
        if not frame.empty:
            if not boot.empty:
                ci_cols = [
                    "trial_data_name",
                    "representation",
                    "score",
                    "estimate",
                    "ci_low",
                    "ci_high",
                ]
                merge_on = ["trial_data_name", "representation", "score"]
                if "application_query" in boot.columns:
                    ci_cols.insert(1, "application_query")
                    merge_on.insert(1, "application_query")
                ci = boot[
                    (boot.get("analysis", "") == "real_vs_simulation")
                    & (boot.get("statistic", "") == "real_vs_simulation_AUROC")
                    & (boot.get("score", "") == "knn")
                ][ci_cols].copy()
                frame = frame.merge(ci, on=merge_on, how="left")
            else:
                frame["estimate"] = frame["auroc_real_vs_simulation"]
                frame["ci_low"] = np.nan
                frame["ci_high"] = np.nan

            if "application_query" in frame.columns:
                labels = [
                    f"{t}\n{a}\n{r}"
                    for t, a, r in zip(
                        frame["trial_data_name"],
                        frame["application_query"],
                        frame["representation"],
                    )
                ]
            else:
                labels = [
                    f"{t}\n{r}"
                    for t, r in zip(
                        frame["trial_data_name"], frame["representation"]
                    )
                ]
            values = frame["auroc_real_vs_simulation"].to_numpy(float)
            fig, ax = plt.subplots(figsize=(max(11, 0.55 * len(frame)), 5.4))
            x = np.arange(len(frame))

            if {"ci_low", "ci_high"} <= set(frame.columns):
                lo = pd.to_numeric(frame["ci_low"], errors="coerce").to_numpy(float)
                hi = pd.to_numeric(frame["ci_high"], errors="coerce").to_numpy(float)
                valid = np.isfinite(lo) & np.isfinite(hi)
                yerr = np.zeros((2, len(frame)), dtype=float)
                yerr[0, valid] = np.maximum(0.0, values[valid] - lo[valid])
                yerr[1, valid] = np.maximum(0.0, hi[valid] - values[valid])
                ax.bar(x, values, yerr=yerr, capsize=3)
            else:
                ax.bar(x, values)

            ax.axhline(0.5, linestyle="--", linewidth=1)
            ax.set_xticks(x, labels=labels, rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.set_ylabel("AUROC (application vs simulation)")
            ax.set_title("Application-vs-simulated separability from latent support")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                figure_dir / "final_real_vs_simulation_auc.png",
                dpi=220,
                bbox_inches="tight",
            )
            figures.append(fig)


    if not knn_frame.empty and "exact_model_accuracy" in knn_frame.columns:
        frame = knn_frame[
            (knn_frame["analysis"] == "recovery")
            & knn_frame["exact_model_accuracy"].notna()
        ].copy()
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(9.5, 5.4))
            for (trial, rep), group in frame.groupby(
                ["trial_data_name", "representation"], sort=False
            ):
                group = group.sort_values("knn_k")
                ax.plot(group["knn_k"], group["exact_model_accuracy"], marker="o", label=f"{trial}: {rep}")
            ax.set_xlabel("k in kNN support")
            ax.set_ylabel("Exact-model recovery")
            ax.set_ylim(0, 1)
            ax.set_title("kNN robustness")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(figure_dir / "robustness_knn_recovery.png", dpi=220, bbox_inches="tight")
            figures.append(fig)

    if figures:
        with PdfPages(figure_dir / "summary_figures.pdf") as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches="tight")
        if SHOW_FIGURES:
            plt.show()
        else:
            for fig in figures:
                plt.close(fig)


def main():
    warnings.simplefilter("default")
    ensure_dir(OUTPUT_ROOT)
    log("Final latent-support analysis")
    log(f"Active config: {ACTIVE_REPRESENTATION_CONFIG['config_name']}")
    log(f"Config file: {ACTIVE_REPRESENTATION_CONFIG['config_file']}")
    log(f"Output root: {OUTPUT_ROOT}")
    log(f"Representations: {ALL_REPRESENTATIONS}")
    log(
        f"Application queries: {APPLICATION_QUERY_NAMES}; "
        f"simulated subsample n={APPLICATION_SIMULATED_N} "
        f"(seed={APPLICATION_SAMPLE_SEED})"
    )
    log(f"Primary scores: {PRIMARY_SCORE_KINDS}")
    log(f"kNN values: {KNN_K_VALUES}; primary k={PRIMARY_K}")
    log(
        "Parallel encoder protocol: train m4--m8; construct frozen-encoder "
        "candidate supports post hoc from m1--m3; evaluate the same application "
        "queries and AUROC/C_max diagnostics."
    )
    log(
        "Split-seed robustness varies reference/calibration/test membership only; "
        "it does not claim neural-network initialization robustness."
    )
    if DISABLE_PARAMETER_MANIFOLD:
        log(
            "Parameter-manifold score disabled in this confirmatory script because "
            "the previous polynomial surface failed validation for several models."
        )

    primary_by_trial = {}
    knn_frames = []
    split_frames = []

    for trial_data_name in TRIAL_DATA_NAMES:
        available, reason = trial_data_available(trial_data_name)
        if not available:
            log(f"\n[skip trial] {trial_data_name}: {reason}")
            RUN_MANIFEST_ROWS.append(
                {
                    "run_type": "primary",
                    "trial_data_name": trial_data_name,
                    "status": "skipped",
                    "reason": reason,
                }
            )
            continue
        try:
            outputs = run_primary_dataset(trial_data_name)
            primary_by_trial[trial_data_name] = outputs
            knn_frames.append(run_knn_sensitivity(trial_data_name, outputs))
            if trial_data_name == PRIMARY_TRIAL_DATA_NAME:
                split_frames.append(
                    run_split_seed_robustness(trial_data_name, outputs)
                )
        except Exception as exc:
            log(f"[ERROR] {trial_data_name}: {exc}")
            log(traceback.format_exc())
            RUN_MANIFEST_ROWS.append(
                {
                    "run_type": "primary",
                    "trial_data_name": trial_data_name,
                    "status": "error",
                    "reason": repr(exc),
                }
            )

    knn_frame = (
        pd.concat([x for x in knn_frames if not x.empty], ignore_index=True)
        if any(not x.empty for x in knn_frames)
        else pd.DataFrame()
    )
    split_frame = (
        pd.concat([x for x in split_frames if not x.empty], ignore_index=True)
        if any(not x.empty for x in split_frames)
        else pd.DataFrame()
    )
    split_agg = aggregate_split_robustness(split_frame)

    aggregate_dir = ensure_dir(OUTPUT_ROOT / "aggregate")
    primary_exports = {
        "primary_generalization_summary.csv": concat_tagged(primary_by_trial, "gen_summary"),
        "primary_recovery_summary.csv": concat_tagged(primary_by_trial, "rec_summary"),
        "primary_family_coverage_summary.csv": concat_tagged(primary_by_trial, "mis_summary"),
        "primary_real_summary.csv": concat_tagged(primary_by_trial, "real_summary"),
        "primary_real_vs_simulation_summary.csv": concat_tagged(
            primary_by_trial, "real_vs_sim_summary"
        ),
        "primary_m4m8_heldout_candidate_recovery_summary.csv": concat_tagged(
            primary_by_trial, "m4m8_rec_summary"
        ),
        "primary_m4m8_application_summary.csv": concat_tagged(
            primary_by_trial, "m4m8_app_summary"
        ),
        "primary_m4m8_real_vs_simulation_summary.csv": concat_tagged(
            primary_by_trial, "m4m8_real_vs_sim_summary"
        ),
        "primary_m4m8_bootstrap_95ci.csv": concat_tagged(
            primary_by_trial, "m4m8_bootstrap"
        ),
        "primary_m4m8_training_family_all_queries_summary.csv": concat_tagged(
            primary_by_trial, "m4m8_family_query_summary"
        ),
        "primary_m4m8_training_family_distance_summary.csv": concat_tagged(
            primary_by_trial, "m4m8_family_distance_summary"
        ),
        "primary_m4m8_training_family_known_recovery_summary.csv": concat_tagged(
            primary_by_trial, "m4m8_family_known_recovery_summary"
        ),
        "primary_m4m8_training_family_real_vs_simulation_summary.csv": concat_tagged(
            primary_by_trial, "m4m8_family_app_vs_sim_summary"
        ),
        "primary_m4m8_training_family_bootstrap_95ci.csv": concat_tagged(
            primary_by_trial, "m4m8_family_bootstrap"
        ),
        "primary_candidate_ood_summary.csv": concat_tagged(primary_by_trial, "ood_summary"),
        "primary_classifier_uncertainty_summary.csv": concat_tagged(primary_by_trial, "uncertainty_summary"),
        "primary_classifier_head_recovery_summary.csv": concat_tagged(primary_by_trial, "head_summary"),
        "primary_bootstrap_95ci.csv": concat_tagged(primary_by_trial, "bootstrap"),
        "knn_robustness_all.csv": knn_frame,
        "split_seed_robustness_all.csv": split_frame,
        "split_seed_robustness_aggregate.csv": split_agg,
        "checkpoint_coverage_all.csv": pd.DataFrame(CHECKPOINT_ROWS).drop_duplicates(),
        "run_manifest.csv": pd.DataFrame(RUN_MANIFEST_ROWS),
    }
    for name, frame in primary_exports.items():
        save_frame(frame, aggregate_dir / name)

    make_summary_figures(primary_by_trial, split_agg, knn_frame)

    config = {
        "active_config": ACTIVE_REPRESENTATION_CONFIG["config_name"],
        "config_file": ACTIVE_REPRESENTATION_CONFIG["config_file"],
        "representation_hparams": ACTIVE_REPRESENTATION_CONFIG[
            "representation_hparams"
        ],
        "trial_data_names": list(TRIAL_DATA_NAMES),
        "primary_trial_data_name": PRIMARY_TRIAL_DATA_NAME,
        "primary_split_seed": PRIMARY_SPLIT_SEED,
        "split_robustness_seeds": list(SPLIT_ROBUSTNESS_SEEDS),
        "representations": list(ALL_REPRESENTATIONS),
        "primary_scores": list(PRIMARY_SCORE_KINDS),
        "primary_k": PRIMARY_K,
        "knn_k_values": list(KNN_K_VALUES),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "parameter_manifold_disabled": DISABLE_PARAMETER_MANIFOLD,
        "candidate_set_ood": {
            "candidates": list(CANDIDATE_OOD_CANDIDATES),
            "id": list(CANDIDATE_OOD_ID),
            "related_unseen": list(CANDIDATE_OOD_RELATED_UNSEEN),
            "novel_family": list(CANDIDATE_OOD_NOVEL_FAMILY),
        },
        "parallel_m4m8_encoder_application": {
            "encoder_train_models": list(M4M8_PARALLEL_TRAIN_MODELS),
            "candidate_support_models": list(
                M4M8_PARALLEL_CANDIDATE_MODELS
            ),
            "application_query_names": list(APPLICATION_QUERY_NAMES),
            "interpretation": (
                "m1--m3 are held out from representation learning and are "
                "used only after freezing the encoder to define candidate "
                "latent supports and a held-out simulation baseline."
            ),
        },
        "m4m8_training_family_support": {
            "families": {
                family_name: list(model_names)
                for family_name, model_names in
                M4M8_TRAINING_FAMILY_MODELS.items()
            },
            "heldout_known_family_queries": dict(
                M4M8_HELDOUT_EXPECTED_FAMILY
            ),
            "application_query_names": list(APPLICATION_QUERY_NAMES),
            "definition": (
                "D_F(z)=min_m D_m(z) within each family; family-level "
                "compatibility is calibrated on pooled validation simulations "
                "from family members. Held-out m4--m8 test subjects are the "
                "in-support baseline."
            ),
        },
        "application_query_names": list(APPLICATION_QUERY_NAMES),
        "application_query_names_active_note": (
            "Per-trial active queries may omit missing m* datasets."
        ),
        "application_simulated_n": APPLICATION_SIMULATED_N,
        "application_sample_seed": APPLICATION_SAMPLE_SEED,
        "notes": [
            "Split-seed robustness does not retrain networks.",
            "True network-seed robustness requires separately trained checkpoints.",
            "Parameter manifold omitted because previous validation R2 was poor for several models.",
            "Application queries (r* and optional m*) receive the same candidate-support family assignment and C_max analysis.",
            "The added m4--m8 training-family analysis compares m1/m2/m3, r1, and m9 directly with option-wise (m4--m7) and attribute-wise (m8) supports.",
            "Family raw support is the minimum member-model distance/NLL and is calibrated at the family level using validation simulations.",
            "For m* application queries, subjects are a random subsample of size APPLICATION_SIMULATED_N.",
            "Real-vs-simulation AUROC uses held-out recovery test subjects as the simulation group and 1-C_max as the anomaly score.",
            "Real-vs-simulation bootstrap resamples simulation and application groups separately and does not train a new classifier.",
            "The added m4--m8 encoder protocol does not retrain networks: it reuses the existing generalization checkpoints.",
            "For the added protocol, m1--m3 simulations do not update encoder weights; they are used only post hoc for candidate support construction/calibration/testing.",
        ],
    }
    (OUTPUT_ROOT / "final_analysis_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "results.txt").write_text(
        "\n".join(REPORT_LINES) + "\n", encoding="utf-8"
    )

    log("\nFinished final latent-support analysis.")
    log(f"Send this folder back for interpretation: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
