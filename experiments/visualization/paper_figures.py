from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import importlib.util
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

try:
    from sklearn.decomposition import PCA
except Exception as exc:
    raise ImportError("scikit-learn is required for the PCA latent-space figure.") from exc


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
STEP05_DIR = PROJECT_ROOT / "experiments" / "latent_support"
ACTIVE_CONFIG = "cfg3"
CFG_NAME = ACTIVE_CONFIG
CFG_DIR = STEP05_DIR / "configs"
CFG_PATH = (
    CFG_DIR
    / f"{CFG_NAME}.py"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "output"
    / "visualization"
    / CFG_NAME
)
MAIN_DIR = OUTPUT_ROOT / "main"
SUPP_DIR = OUTPUT_ROOT / "supplementary"
SOURCE_DIR = OUTPUT_ROOT / "source_data"
CACHE_DIR = OUTPUT_ROOT / "cache"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from choice_generators.model_name_transfer import (
    paper_data_label,
)


def figure_model_label(data_name: str) -> str:
    return paper_data_label(data_name)


TRIAL_DATA_NAMES = (
    "pub_Konstantinidis2020",
    "pub_Thrailkill2023",
)
TRIAL_LABELS = {
    "pub_Konstantinidis2020": "380-trial design",
    "pub_Thrailkill2023": "27-trial design",
}
TRIAL_SHORT = {
    "pub_Konstantinidis2020": "380 trials",
    "pub_Thrailkill2023": "27 trials",
}

REPRESENTATION_TRAIN_TRIAL_DATA_NAME = "general_400"
TRAIN_MODELS = ("m4", "m5", "m6", "m7", "m8")
HELDOUT_POINTER_MODELS = ("m1", "m3")
NOVEL_HEURISTIC = ("m9",)
REAL_QUERY = ("r1",)


PRIMARY_SCORE = "knn"
PRIMARY_BEHAVIOR_DISTANCE = "functional_rmse"
MODEL_RECOVERY_KNN_K = 10
SEED = 31


MAIN_REPRESENTATIONS = (
    "nn1",
    "classifier",
    "nn1c",
    "parameter_decoder",
)
ALL_REPRESENTATIONS = (
    "nn1",
    "nn1b",
    "classifier",
    "nn1c",
    "nn1bc",
    "parameter_decoder",
)

REP_LABEL = {
    "nn1": "NP",
    "nn1b": "NP + consistency",
    "classifier": "Classifier",
    "nn1c": "NP + classifier",
    "nn1bc": "NP + consistency + classifier",
    "parameter_decoder": "Parameter decoder",
}

PCA_PLOT_CAP_PER_SIMULATOR = 300
PCA_PLOT_CAP_REAL = 300
DPI = 450
SHOW_FIGURES = True

SHOW_LATENT_MODEL_LABELS = 0


JOINT_DESIGN_MAIN_TRIAL = "general_400"
JOINT_DESIGN_MAIN_MODELS = ("m1", "m3", "m4", "m5", "m6", "m7", "m8")
JOINT_DESIGN_EXTRA_MODELS = ("m1", "m3")
JOINT_DESIGN_EXTRA_TRIALS = (
    "pub_Konstantinidis2020",
    "pub_Thrailkill2023",
)
JOINT_DESIGN_SOURCES = (
    JOINT_DESIGN_MAIN_TRIAL,
    *JOINT_DESIGN_EXTRA_TRIALS,
)
JOINT_DESIGN_MODELS = JOINT_DESIGN_MAIN_MODELS
JOINT_DESIGN_MARKERS = {
    "general_400": "o",
    "pub_Konstantinidis2020": "^",
    "pub_Thrailkill2023": "X",
}
JOINT_DESIGN_MAIN_CAP_PER_MODEL = 250
JOINT_DESIGN_EXTRA_CAP_PER_MODEL = 80


POSTERIOR_DIAGNOSTIC_MODELS = (
    "m1", "m3", "m4", "m5", "m6", "m7", "m8",
)
POSTERIOR_REPRESENTATION = "nn1"
POSTERIOR_DESIGNS = (
    "general_400",
    "pub_Konstantinidis2020",
    "pub_Thrailkill2023",
)
POSTERIOR_DESIGN_LABELS = {
    "general_400": "Full trial-space",
    "pub_Konstantinidis2020": "380 trials",
    "pub_Thrailkill2023": "27 trials",
}
POSTERIOR_PLOT_ORDER = (
    "general_400",
    "pub_Konstantinidis2020",
    "pub_Thrailkill2023",
)


OPTION_WISE = {"m1", "m4", "m5", "m6", "m7"}
ATTRIBUTE_WISE = {"m3", "m8"}


COLORS = {
    "option_wise": "#0072B2",
    "attribute_wise": "#D55E00",
    "heuristic": "#CC79A7",
    "real": "#222222",
    "np": "#0072B2",
    "classifier": "#D55E00",
    "design_380": "#0072B2",
    "design_27": "#D55E00",
    "training_sim": "#777777",
}
REP_COLORS = {
    "nn1": "#0072B2",
    "nn1b": "#56B4E9",
    "classifier": "#D55E00",
    "nn1c": "#009E73",
    "nn1bc": "#CC79A7",
    "parameter_decoder": "#E69F00",
}


FAMILY_MARKERS = {
    "option_wise": "o",
    "attribute_wise": "^",
    "heuristic": "X",
    "real": "x",
}

MANIFEST_ROWS: list[dict] = []


def ensure_dirs():
    for path in (OUTPUT_ROOT, MAIN_DIR, SUPP_DIR, SOURCE_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "figure.titlesize": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def annotate_panel(ax, letter: str):
    ax.text(
        -0.12,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(
    fig,
    stem: str,
    role: str,
    description: str,
    bundle=None,
):
    del bundle
    print(f"Showing {role} figure: {stem}")
    print(f"  {description}")
    plt.show()
    plt.close(fig)


def write_source(frame: pd.DataFrame, filename: str):
    path = SOURCE_DIR / filename
    frame.to_csv(path, index=False)
    return path


def _fmt(value, digits=3):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def print_figure_header(title: str):
    print("")
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_figure_02_results(source: pd.DataFrame):
    print_figure_header("Figure 2 numeric summary")
    plot_order = ["m4", "m5", "m6", "m7", "m8", "m1", "m3", "m9", "r1"]
    for trial in TRIAL_DATA_NAMES:
        part = source[source["trial_data_name"] == trial]
        print(f"{TRIAL_LABELS[trial]}")
        if part.empty:
            print("  (no points)")
            continue
        ev1 = float(part["pc1_explained_variance"].iloc[0])
        ev2 = float(part["pc2_explained_variance"].iloc[0])
        print(
            f"  PCA variance: PC1={100 * ev1:.1f}%, "
            f"PC2={100 * ev2:.1f}%"
        )
        for name in plot_order:
            sub = part[part["data_name"] == name]
            if sub.empty:
                continue
            print(
                f"  {figure_model_label(name):<6}  n={len(sub):4d}  "
                f"status={sub['status'].iloc[0]:<8}  "
                f"PC1 median={_fmt(sub['pc1'].median())}  "
                f"PC2 median={_fmt(sub['pc2'].median())}"
            )


def print_figure_03_results(delta: pd.DataFrame):
    print_figure_header("Appendix Figure S1 numeric summary")
    model_order = ["m4", "m5", "m6", "m7", "m8", "m1", "m3", "m9"]
    for trial in TRIAL_DATA_NAMES:
        print(f"{TRIAL_LABELS[trial]}")
        sub = delta[delta["trial_data_name"] == trial].set_index("model")
        for model in model_order:
            if model not in sub.index:
                continue
            row = sub.loc[model]
            print(
                f"  {figure_model_label(model):<6}  "
                f"rho_NP={_fmt(row['rho_np'])}  "
                f"rho_classifier={_fmt(row['rho_classifier'])}  "
                f"delta={_fmt(row['delta_rho'])}  "
                f"({row['status']})"
            )


def print_figure_04_results(cmax: pd.DataFrame, boot: pd.DataFrame):
    print_figure_header("Figure 4 numeric summary")
    groups = ["Training simulations", "Held-out H1", "Real data"]
    print("C_max (median, mean, fraction < 0.05)")
    for trial in TRIAL_DATA_NAMES:
        print(f"  {TRIAL_LABELS[trial]}")
        part = cmax[cmax["trial_data_name"] == trial]
        for group in groups:
            values = part.loc[part["plot_group"] == group, "c_max"].dropna()
            if values.empty:
                print(f"    {group}: (no values)")
                continue
            frac = float((values < 0.05).mean())
            print(
                f"    {group:<22}  n={len(values):4d}  "
                f"median={_fmt(values.median())}  "
                f"mean={_fmt(values.mean())}  "
                f"P(C_max<0.05)={_fmt(frac)}"
            )
    print("AUROC (1 - C_max) with 95% CI")
    reps = [
        rep for rep in MAIN_REPRESENTATIONS
        if rep in set(boot["representation"])
    ]
    for query, title in (("r1", "Real vs training sims"), ("m9", "H1 vs training sims")):
        print(f"  {title}")
        for trial in TRIAL_DATA_NAMES:
            print(f"    {TRIAL_LABELS[trial]}")
            for rep in reps:
                row = boot[
                    (boot["trial_data_name"] == trial)
                    & (boot["application_query"] == query)
                    & (boot["representation"] == rep)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                print(
                    f"      {REP_LABEL[rep]:<32}  "
                    f"AUROC={_fmt(row['estimate'])}  "
                    f"95% CI [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]"
                )


def print_figure_s1_results(summary: pd.DataFrame):
    print_figure_header("Figure S1 numeric summary")
    for trial in TRIAL_DATA_NAMES:
        print(f"{TRIAL_LABELS[trial]}")
        sub = summary[summary["trial_data_name"] == trial]
        reps = [
            rep for rep in ALL_REPRESENTATIONS
            if rep in set(sub["representation"])
        ]
        for group in ("Training simulators", "Held-out OW1/AW1"):
            print(f"  {group}")
            vals = sub[sub["group"] == group].set_index("representation")
            for rep in reps:
                if rep not in vals.index:
                    continue
                print(
                    f"    {REP_LABEL[rep]:<32}  "
                    f"mean within-model rho={_fmt(vals.loc[rep, 'mean_within_model_rho'])}"
                )


def print_figure_s2_results(frame: pd.DataFrame):
    print_figure_header("Appendix Figure S3 numeric summary")
    model_cols = [("accuracy_m1", "OW1"), ("accuracy_m3", "AW1")]
    for trial in TRIAL_DATA_NAMES:
        print(f"{TRIAL_LABELS[trial]}")
        sub = frame[frame["trial_data_name"] == trial]
        reps = [
            rep for rep in MAIN_REPRESENTATIONS
            if rep in set(sub["representation"])
        ]
        for rep in reps:
            row = sub[sub["representation"] == rep]
            if row.empty:
                continue
            row = row.iloc[0]
            bits = [
                f"{label}={_fmt(row[col])}"
                for col, label in model_cols
            ]
            print(f"  {REP_LABEL[rep]:<32}  " + "  ".join(bits))


def print_figure_s3_results(frame: pd.DataFrame):
    print_figure_header("Figure S3 numeric summary")
    score_order = ["mean_euclidean", "knn", "gaussian_nll"]
    score_labels = {
        "mean_euclidean": "Mean Euclidean",
        "knn": "kNN",
        "gaussian_nll": "Gaussian NLL",
    }
    for query, title in (("r1", "Real data"), ("m9", "Held-out H1")):
        print(f"{title}")
        for trial in TRIAL_DATA_NAMES:
            print(f"  {TRIAL_LABELS[trial]}")
            sub = frame[
                (frame["trial_data_name"] == trial)
                & (frame["application_query"] == query)
            ].set_index("score")
            for score in score_order:
                if score not in sub.index:
                    continue
                print(
                    f"    {score_labels[score]:<16}  "
                    f"AUROC={_fmt(sub.loc[score, 'auroc_real_vs_simulation'])}"
                )


def print_figure_s4_results(frame: pd.DataFrame):
    print_figure_header("Appendix Figure S5 numeric summary")
    pair_order = [
        "within_option_m4",
        "training_cross_family_m4_m8",
        "heldout_known_m1_m3",
        "heldout_to_heuristic_m1_m9",
    ]
    pair_labels = {
        "within_option_m4": "OW2→OW2",
        "training_cross_family_m4_m8": "OW2→AW2",
        "heldout_known_m1_m3": "OW1→AW1",
        "heldout_to_heuristic_m1_m9": "OW1→H1",
    }
    for trial in TRIAL_DATA_NAMES:
        print(f"{TRIAL_LABELS[trial]}")
        sub = frame[frame["trial_data_name"] == trial].set_index("pair")
        for pair in pair_order:
            if pair not in sub.index:
                continue
            true_y = float(sub.loc[pair, "true_endpoint_distance"])
            dec_y = float(sub.loc[pair, "decoded_endpoint_distance"])
            print(
                f"  {pair_labels[pair]:<10}  "
                f"true={_fmt(true_y)}  "
                f"NP-decoded={_fmt(dec_y)}  "
                f"decoded/true={_fmt(dec_y / true_y if true_y else np.nan)}"
            )


def print_figure_s5_results(
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
):
    print_figure_header("Appendix Figure S2 numeric summary")
    print(
        "Five-way recovery among representation-training simulators: "
        "OW2, OW3, OW4, OW5, AW2"
    )
    print(f"Chance exact-model accuracy = {1 / len(TRAIN_MODELS):.3f}")

    for trial in TRIAL_DATA_NAMES:
        print("")
        print(f"{TRIAL_LABELS[trial]}")
        sub = summary[summary["trial_data_name"] == trial]
        reps = [
            rep for rep in MAIN_REPRESENTATIONS
            if rep in set(sub["representation"])
        ]

        for rep in reps:
            overall = sub[
                (sub["representation"] == rep)
                & (sub["true_model"] == "__overall__")
            ]
            if overall.empty:
                continue

            overall_acc = float(overall.iloc[0]["exact_model_accuracy"])
            print(
                f"  {REP_LABEL[rep]:<32}  "
                f"overall exact recovery={_fmt(overall_acc)}"
            )

            model_rows = sub[
                (sub["representation"] == rep)
                & (sub["true_model"] != "__overall__")
            ].set_index("true_model")

            bits = []
            for model in TRAIN_MODELS:
                if model not in model_rows.index:
                    continue
                bits.append(
                    f"{figure_model_label(model)}="
                    f"{_fmt(model_rows.loc[model, 'exact_model_accuracy'])}"
                )
            if bits:
                print("    Per-model accuracy: " + "  ".join(bits))


            conf = confusion[
                (confusion["trial_data_name"] == trial)
                & (confusion["representation"] == rep)
            ].copy()
            if conf.empty:
                continue

            matrix = conf.pivot_table(
                index="true_model",
                columns="predicted_model",
                values="proportion",
                aggfunc="first",
                fill_value=0.0,
            ).reindex(
                index=list(TRAIN_MODELS),
                columns=list(TRAIN_MODELS),
                fill_value=0.0,
            )

            paper_labels = [figure_model_label(m) for m in TRAIN_MODELS]
            print("    Normalized confusion matrix (rows=true, cols=predicted):")
            header = " " * 14 + "".join(f"{lab:>8}" for lab in paper_labels)
            print(header)
            for true_model, row_label in zip(TRAIN_MODELS, paper_labels, strict=True):
                values = "".join(
                    f"{float(matrix.loc[true_model, pred_model]):8.3f}"
                    for pred_model in TRAIN_MODELS
                )
                print(f"      {row_label:<8}{values}")


def family_for_model(name: str) -> str:
    if name in OPTION_WISE:
        return "option_wise"
    if name in ATTRIBUTE_WISE:
        return "attribute_wise"
    if name == "m9":
        return "heuristic"
    if name.startswith("r"):
        return "real"
    return "unknown"


def find_support_root() -> Path:
    candidates = [
        PROJECT_ROOT
        / "output"
        / "latent_support_final_analysis"
        / "parallel_m4m8"
        / CFG_NAME,
        PROJECT_ROOT
        / "output"
        / "latent_support_final_analysis"
        / CFG_NAME,
    ]
    required = (
        "aggregate/primary_m4m8_training_family_real_vs_simulation_summary.csv"
    )
    for root in candidates:
        if (root / required).exists():
            return root
    message = "\n".join(f"  - {x}" for x in candidates)
    raise FileNotFoundError(
        f"Could not locate {CFG_NAME} latent-support outputs. Tried:\n"
        f"{message}\n"
        "Run the latest step_03 parallel m4--m8 family-support analysis first."
    )


def geometry_root() -> Path:
    root = PROJECT_ROOT / "output" / "latent_geometry_validation" / "aggregate"
    return root


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"Optional figure input missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def jitter_points(values, n_max=180, seed=SEED, width=0.055):
    values = np.asarray(values, dtype=float)
    if len(values) > n_max:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(values), size=n_max, replace=False)
        values = values[keep]
    rng = np.random.default_rng(seed + 17)
    x = rng.uniform(-width, width, len(values))
    return x, values


_CORE = None
_CFG = None


def load_core_config():
    global _CORE, _CFG
    if _CORE is not None:
        return _CORE, _CFG

    src_dir = PROJECT_ROOT / "src"
    latent_dir = PROJECT_ROOT / "experiments" / "representation_learning"
    for path in (PROJECT_ROOT, src_dir, latent_dir, STEP05_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    core_path = STEP05_DIR / "support_core.py"
    core = load_module(core_path, "step06_latent_support_core")
    if not CFG_PATH.exists():
        available = sorted(path.stem for path in CFG_DIR.glob("cfg*.py"))
        raise FileNotFoundError(
            f"Missing config file: {CFG_PATH}. Available: {available}"
        )
    cfg = load_module(CFG_PATH, f"step06_{CFG_NAME}")

    if getattr(cfg, "CONFIG_NAME", None) != CFG_NAME:
        raise ValueError(
            f"Expected CONFIG_NAME={CFG_NAME!r}, got "
            f"{getattr(cfg, 'CONFIG_NAME', None)!r}"
        )


    core.REPRESENTATION_TRAIN_TRIAL_DATA_NAME = (
        REPRESENTATION_TRAIN_TRIAL_DATA_NAME
    )
    core.REPRESENTATION_HPARAMS = dict(cfg.REPRESENTATION_HPARAMS)
    core.MAX_POINTS_PER_DATA = 1000

    _CORE, _CFG = core, cfg
    return core, cfg


def test_indices_for_group(experiment, name: str) -> np.ndarray:
    group = experiment.subject_indices_by_data[name].cpu().numpy().astype(np.int64)
    split = experiment.subject_splits["test"].cpu().numpy().astype(np.int64)
    selected = group[np.isin(group, split)]
    return selected


def capped_indices(
    values: np.ndarray,
    cap: int,
    trial_data_name: str,
    group: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if cap <= 0 or len(values) <= cap:
        return values

    seed = SEED + sum(ord(ch) for ch in f"{trial_data_name}-{group}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(values, size=cap, replace=False))


def build_latent_map_frame(trial_data_name: str, force=False) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"latent_map_{trial_data_name}_{CFG_NAME}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    core, _ = load_core_config()
    data_names = (
        "r1",
        "m1",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
        "m8",
        "m9",
    )
    experiment = core.load_experiment_data(
        trial_data_name,
        data_names=data_names,
        seed=SEED,
        full_real=True,
    )
    rep = core.load_representation("nn1", TRAIN_MODELS)
    if rep is None:
        raise FileNotFoundError(
            f"{CFG_NAME} NP checkpoint for general_400 / m4--m8 was not found."
        )
    latent, _ = core.encode_representation(rep, experiment)
    scale = core.fit_latent_scale(latent, experiment, TRAIN_MODELS)
    z = scale.transform(latent)


    pca_fit_idx = np.concatenate(
        [
            np.asarray(
                core.indices_for_data_split(experiment, model, "train"),
                dtype=np.int64,
            )
            for model in TRAIN_MODELS
        ]
    )
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(z[pca_fit_idx])

    rows = []
    for name in data_names:
        idx = test_indices_for_group(experiment, name)
        cap = PCA_PLOT_CAP_REAL if name.startswith("r") else PCA_PLOT_CAP_PER_SIMULATOR
        idx = capped_indices(idx, cap, trial_data_name, name)
        if len(idx) == 0:
            continue
        xy = pca.transform(z[idx])
        status = (
            "training"
            if name in TRAIN_MODELS
            else ("real" if name.startswith("r") else "heldout")
        )
        family = family_for_model(name)
        for subject_index, point in zip(idx, xy, strict=True):
            rows.append(
                {
                    "trial_data_name": trial_data_name,
                    "data_name": name,
                    "family": family,
                    "status": status,
                    "combined_subject_index": int(subject_index),
                    "pc1": float(point[0]),
                    "pc2": float(point[1]),
                    "pc1_explained_variance": float(
                        pca.explained_variance_ratio_[0]
                    ),
                    "pc2_explained_variance": float(
                        pca.explained_variance_ratio_[1]
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return frame


def draw_latent_panel(ax, frame: pd.DataFrame, trial_data_name: str):
    order = ["m4", "m5", "m6", "m7", "m8", "m1", "m3", "m9", "r1"]

    for model in order:
        part = frame[frame["data_name"] == model]
        if part.empty:
            continue
        family = family_for_model(model)
        color = COLORS[family]
        marker = FAMILY_MARKERS[family]
        status = part["status"].iloc[0]

        if status == "training":
            ax.scatter(
                part["pc1"],
                part["pc2"],
                s=10,
                marker=marker,
                facecolors=color,
                edgecolors="none",
                alpha=0.28,
                rasterized=True,
                zorder=1,
            )
        elif status == "heldout":

            if model == "m9":
                ax.scatter(
                    part["pc1"],
                    part["pc2"],
                    s=18,
                    marker=marker,
                    c=color,
                    edgecolors="white",
                    linewidths=0.25,
                    alpha=0.55,
                    rasterized=True,
                    zorder=3,
                )
            else:
                ax.scatter(
                    part["pc1"],
                    part["pc2"],
                    s=15,
                    marker=marker,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.65,
                    alpha=0.45,
                    rasterized=True,
                    zorder=2,
                )
        else:
            ax.scatter(
                part["pc1"],
                part["pc2"],
                s=28,
                marker="x",
                c=COLORS["real"],
                linewidths=0.9,
                alpha=0.95,
                rasterized=True,
                zorder=5,
            )


        if SHOW_LATENT_MODEL_LABELS and not model.startswith("r"):
            cx = float(part["pc1"].median())
            cy = float(part["pc2"].median())
            label_offsets = {
                "m1": (-9, 8),
                "m3": (-8, -8),
                "m4": (-10, 7),
                "m5": (9, 7),
                "m6": (-9, -8),
                "m7": (9, -7),
                "m8": (8, 8),
                "m9": (0, 9),
            }
            dx, dy = label_offsets.get(model, (0, 0))
            ax.annotate(
                figure_model_label(model),
                xy=(cx, cy),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5.9,
                fontweight="bold",
                color=(color if model != "m9" else COLORS["heuristic"]),
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.72,
                ),
                zorder=6,
            )

    ev1 = frame["pc1_explained_variance"].iloc[0]
    ev2 = frame["pc2_explained_variance"].iloc[0]
    ax.set_xlabel(f"PC1 ({100 * ev1:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * ev2:.1f}%)")
    ax.set_title(TRIAL_LABELS[trial_data_name])


def make_figure_02_latent_geometry(force_cache=False):
    frames = []
    for trial in TRIAL_DATA_NAMES:
        frames.append(build_latent_map_frame(trial, force=force_cache))
    source = pd.concat(frames, ignore_index=True)
    write_source(source, "figure_02_shared_latent_geometry.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    for letter, trial, ax in zip("AB", TRIAL_DATA_NAMES, axes, strict=True):
        draw_latent_panel(
            ax,
            source[source["trial_data_name"] == trial],
            trial,
        )
        annotate_panel(ax, letter)

    family_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=COLORS["option_wise"],
            markeredgecolor=COLORS["option_wise"],
            label="OW training models (OW2–OW5)",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="none", markersize=5,
            markerfacecolor=COLORS["attribute_wise"],
            markeredgecolor=COLORS["attribute_wise"],
            label="AW training model (AW2)",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor="white",
            markeredgecolor=COLORS["option_wise"],
            label="Held-out OW1",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="none", markersize=5,
            markerfacecolor="white",
            markeredgecolor=COLORS["attribute_wise"],
            label="Held-out AW1",
        ),
        Line2D(
            [0], [0], marker="X", linestyle="none", markersize=5,
            markerfacecolor=COLORS["heuristic"],
            markeredgecolor=COLORS["heuristic"],
            label="Held-out H1",
        ),
        Line2D(
            [0], [0], marker="x", linestyle="none", markersize=5,
            color=COLORS["real"], label="Real participants",
        ),
    ]
    fig.legend(
        handles=family_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.25, wspace=0.25)
    print_figure_02_results(source)
    return fig


def build_joint_design_latent_frame(force=False) -> pd.DataFrame:
    cache_path = (
        CACHE_DIR
        / f"joint_design_latent_map_general_extra380_27_sparse_{CFG_NAME}.csv"
    )
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    core, _ = load_core_config()
    rep = core.load_representation("nn1", TRAIN_MODELS)
    if rep is None:
        raise FileNotFoundError(
            f"{CFG_NAME} NP checkpoint for general_400 / m4--m8 was not found."
        )


    main_experiment = core.load_experiment_data(
        JOINT_DESIGN_MAIN_TRIAL,
        data_names=JOINT_DESIGN_MAIN_MODELS,
        seed=SEED,
        full_real=False,
    )
    main_latent, _ = core.encode_representation(rep, main_experiment)
    main_latent = np.asarray(main_latent, dtype=float)
    sources = []
    for model in JOINT_DESIGN_MAIN_MODELS:
        indices = capped_indices(
            test_indices_for_group(main_experiment, model),
            JOINT_DESIGN_MAIN_CAP_PER_MODEL,
            JOINT_DESIGN_MAIN_TRIAL,
            f"joint-{model}",
        )
        if len(indices):
            sources.append(
                {
                    "trial": JOINT_DESIGN_MAIN_TRIAL,
                    "model": model,
                    "latent": main_latent[indices],
                    "indices": indices,
                }
            )


    for trial in JOINT_DESIGN_EXTRA_TRIALS:
        for model in JOINT_DESIGN_EXTRA_MODELS:
            extra_experiment = core.load_experiment_data(
                trial,
                data_names=(model,),
                seed=SEED,
                full_real=False,
            )
            extra_latent, _ = core.encode_representation(rep, extra_experiment)
            extra_latent = np.asarray(extra_latent, dtype=float)
            indices = capped_indices(
                test_indices_for_group(extra_experiment, model),
                JOINT_DESIGN_EXTRA_CAP_PER_MODEL,
                trial,
                f"joint-{model}",
            )
            if len(indices):
                sources.append(
                    {
                        "trial": trial,
                        "model": model,
                        "latent": extra_latent[indices],
                        "indices": indices,
                    }
                )

    if not sources:
        raise RuntimeError("No latent points were available for the joint PCA.")


    embedding_latents = np.vstack([source["latent"] for source in sources])
    pca = PCA(n_components=2, random_state=SEED)
    all_xy = pca.fit_transform(embedding_latents)

    rows = []
    start = 0
    for source in sources:
        stop = start + len(source["latent"])
        xy = all_xy[start:stop]
        start = stop
        model = source["model"]
        for subject_index, point in zip(
            source["indices"],
            xy,
            strict=True,
        ):
            rows.append(
                {
                    "trial_data_name": source["trial"],
                    "data_name": model,
                    "paper_label": figure_model_label(model),
                    "family": family_for_model(model),
                    "combined_subject_index": int(subject_index),
                    "pc1": float(point[0]),
                    "pc2": float(point[1]),
                    "pc1_explained_variance": float(
                        pca.explained_variance_ratio_[0]
                    ),
                    "pc2_explained_variance": float(
                        pca.explained_variance_ratio_[1]
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return frame


def print_joint_design_latent_results(source: pd.DataFrame):
    print_figure_header("MAIN Figure 4 numeric summary")
    if source.empty:
        print("(no points)")
        return

    ev1 = float(source["pc1_explained_variance"].iloc[0])
    ev2 = float(source["pc2_explained_variance"].iloc[0])
    print(
        "general_400 main data; 380/27-trial OW1/AW1 EXTRA_DATA; "
        "raw latent PCA fit together: "
        f"PC1={100 * ev1:.1f}%, PC2={100 * ev2:.1f}%"
    )
    source_labels = {
        "general_400": "general_400",
        **TRIAL_LABELS,
    }
    for trial in JOINT_DESIGN_SOURCES:
        print(source_labels[trial])
        part = source[source["trial_data_name"] == trial]
        models = (
            JOINT_DESIGN_MAIN_MODELS
            if trial == JOINT_DESIGN_MAIN_TRIAL
            else JOINT_DESIGN_EXTRA_MODELS
        )
        for model in models:
            sub = part[part["data_name"] == model]
            if sub.empty:
                continue
            print(
                f"  {figure_model_label(model):<6}  n={len(sub):4d}  "
                f"PC1 median={_fmt(sub['pc1'].median())}  "
                f"PC2 median={_fmt(sub['pc2'].median())}"
            )


def make_main_figure_03_joint_design_latent(force_cache=False):
    source = build_joint_design_latent_frame(force=force_cache)
    write_source(source, "main_figure_03_joint_design_latent.csv")

    fig, ax = plt.subplots(1, 1, figsize=(5.35, 4.15))


    for trial in JOINT_DESIGN_SOURCES:
        marker = JOINT_DESIGN_MARKERS[trial]
        for family in ("option_wise", "attribute_wise"):
            part = source[
                (source["trial_data_name"] == trial)
                & (source["family"] == family)
            ]
            if part.empty:
                continue

            color = COLORS[family]

            if trial == "pub_Konstantinidis2020":
                ax.scatter(
                    part["pc1"],
                    part["pc2"],
                    s=24,
                    marker=marker,
                    facecolors=color,
                    edgecolors="white",
                    linewidths=0.25,
                    alpha=0.78,
                    rasterized=True,
                    zorder=4,
                )
            elif trial == "pub_Thrailkill2023":
                ax.scatter(
                    part["pc1"],
                    part["pc2"],
                    s=28,
                    marker=marker,
                    c=color,
                    linewidths=0.75,
                    alpha=0.88,
                    rasterized=True,
                    zorder=5,
                )
            else:
                ax.scatter(
                    part["pc1"],
                    part["pc2"],
                    s=9,
                    marker=marker,
                    facecolors=color,
                    edgecolors="none",
                    linewidths=0.0,
                    alpha=0.16,
                    rasterized=True,
                    zorder=1,
                )

    ev1 = float(source["pc1_explained_variance"].iloc[0])
    ev2 = float(source["pc2_explained_variance"].iloc[0])
    ax.set_xlabel(f"PC1 ({100 * ev1:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * ev2:.1f}%)")
    ax.set_title("Shared latent space with published-design overlays")


    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor=COLORS["option_wise"],
            markeredgecolor=COLORS["option_wise"],
            label="OW family",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor=COLORS["attribute_wise"],
            markeredgecolor=COLORS["attribute_wise"],
            label="AW family",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor="0.55",
            markeredgecolor="none",
            alpha=0.35,
            label="Full trial-space",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="none", markersize=5.8,
            markerfacecolor="0.30",
            markeredgecolor="0.30",
            label="380-trial design",
        ),
        Line2D(
            [0], [0], marker="X", linestyle="none", markersize=5.8,
            color="0.30",
            label="27-trial design",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="best",
    )

    fig.subplots_adjust(bottom=0.14)
    print_joint_design_latent_results(source)
    return fig


def draw_joint_design_latent_panel(
    ax,
    source: pd.DataFrame,
    title: str = "Joint projection across designs",
):
    for trial in JOINT_DESIGN_SOURCES:
        marker = JOINT_DESIGN_MARKERS[trial]
        for family in ("option_wise", "attribute_wise"):
            part = source[(source["trial_data_name"] == trial) & (source["family"] == family)]
            if part.empty:
                continue
            color = COLORS[family]
            if trial == "pub_Konstantinidis2020":
                ax.scatter(part["pc1"], part["pc2"], s=23, marker=marker,
                           facecolors=color, edgecolors="white", linewidths=0.25,
                           alpha=0.78, rasterized=True, zorder=4)
            elif trial == "pub_Thrailkill2023":
                ax.scatter(part["pc1"], part["pc2"], s=27, marker=marker, c=color,
                           linewidths=0.45, alpha=0.88, rasterized=True, zorder=5)
            else:
                ax.scatter(part["pc1"], part["pc2"], s=8.5, marker=marker,
                           facecolors=color, edgecolors="none", linewidths=0.0,
                           alpha=0.16, rasterized=True, zorder=1)

    ev1 = float(source["pc1_explained_variance"].iloc[0])
    ev2 = float(source["pc2_explained_variance"].iloc[0])
    ax.set_xlabel(f"PC1 ({100 * ev1:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * ev2:.1f}%)")
    ax.set_title(title)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.2,
               markerfacecolor=COLORS["option_wise"], markeredgecolor=COLORS["option_wise"],
               label="OW family"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.2,
               markerfacecolor=COLORS["attribute_wise"], markeredgecolor=COLORS["attribute_wise"],
               label="AW family"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.2,
               markerfacecolor="0.55", markeredgecolor="none", alpha=0.35,
               label="Full trial-space"),
        Line2D([0], [0], marker="^", linestyle="none", markersize=5.5,
               markerfacecolor="0.30", markeredgecolor="0.30",
               label="380-trial design"),
        Line2D([0], [0], marker="X", linestyle="none", markersize=5.5,
               color="0.30", label="27-trial design"),
    ]


def make_main_figure_02_combined_latent_geometry(force_cache=False):
    design_frames = [
        build_latent_map_frame(trial, force=force_cache)
        for trial in TRIAL_DATA_NAMES
    ]
    design_source = pd.concat(design_frames, ignore_index=True)
    joint_source = build_joint_design_latent_frame(force=force_cache)


    write_source(design_source, "figure_02_shared_latent_geometry.csv")
    write_source(joint_source, "main_figure_03_joint_design_latent.csv")
    write_source(design_source, "figure_02_combined_latent_geometry_design_specific.csv")
    write_source(joint_source, "figure_02_combined_latent_geometry_cross_design.csv")


    fig = plt.figure(figsize=(7.35, 3.05))
    gs = fig.add_gridspec(
        1,
        3,
        wspace=0.24,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    for letter, trial, ax in zip("AB", TRIAL_DATA_NAMES, (ax_a, ax_b), strict=True):
        draw_latent_panel(ax, design_source[design_source["trial_data_name"] == trial], trial)
        annotate_panel(ax, letter)

    family_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=COLORS["option_wise"], markeredgecolor=COLORS["option_wise"],
               label="OW training models (OW2–OW5)"),
        Line2D([0], [0], marker="^", linestyle="none", markersize=5,
               markerfacecolor=COLORS["attribute_wise"], markeredgecolor=COLORS["attribute_wise"],
               label="AW training model (AW2)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor="white", markeredgecolor=COLORS["option_wise"],
               label="Held-out OW1"),
        Line2D([0], [0], marker="^", linestyle="none", markersize=5,
               markerfacecolor="white", markeredgecolor=COLORS["attribute_wise"],
               label="Held-out AW1"),
        Line2D([0], [0], marker="X", linestyle="none", markersize=5,
               markerfacecolor=COLORS["heuristic"], markeredgecolor=COLORS["heuristic"],
               label="Held-out H1"),
        Line2D([0], [0], marker="x", linestyle="none", markersize=5,
               color=COLORS["real"], label="Real participants"),
    ]


    fig.legend(
        handles=family_handles,
        loc="center",
        bbox_to_anchor=(0.335, 0.165),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        columnspacing=0.95,
        handletextpad=0.45,
    )

    joint_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.2,
            markerfacecolor=COLORS["option_wise"],
            markeredgecolor=COLORS["option_wise"],
            label="OW family",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.2,
            markerfacecolor=COLORS["attribute_wise"],
            markeredgecolor=COLORS["attribute_wise"],
            label="AW family",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.2,
            markerfacecolor="0.55",
            markeredgecolor="none",
            alpha=0.35,
            label="Full trial-space",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="none", markersize=5.5,
            markerfacecolor="0.30",
            markeredgecolor="0.30",
            label="380-trial design",
        ),
        Line2D(
            [0], [0], marker="X", linestyle="none", markersize=5.5,
            color="0.30",
            markeredgewidth=0.45,
            label="27-trial design",
        ),
    ]


    blank_handle = Line2D(
        [],
        [],
        linestyle="none",
        marker="",
        alpha=0.0,
        label="",
    )
    joint_legend_handles = [
        joint_handles[0],
        joint_handles[1],
        blank_handle,
        joint_handles[2],
        joint_handles[3],
        joint_handles[4],
    ]
    fig.legend(
        handles=joint_legend_handles,
        loc="center",
        bbox_to_anchor=(0.835, 0.165),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        columnspacing=0.90,
        handletextpad=0.45,
    )

    draw_joint_design_latent_panel(
        ax_c, joint_source,
        title="Joint projection across designs",
    )
    annotate_panel(ax_c, "C")
    for ax in (ax_a, ax_b, ax_c):
        ax.title.set_fontsize(10.0)


    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.34,
        top=0.91,
    )
    print_figure_02_results(design_source)
    print_joint_design_latent_results(joint_source)
    return fig


def load_geometry_table():
    root = geometry_root()
    return require_csv(root / "geometry_fidelity_summary.csv")


def draw_within_model_geometry(ax, geometry: pd.DataFrame, trial: str):

    model_order = ["m4", "m5", "m6", "m7", "m8", "m1", "m3", "m9"]
    part = geometry[
        (geometry["trial_data_name"] == trial)
        & (geometry["protocol"] == "generalization_m4_m8")
        & (geometry["scope"] == "within_model")
        & (geometry["behavior_distance"] == PRIMARY_BEHAVIOR_DISTANCE)
        & (geometry["representation"].isin(["nn1", "classifier"]))
    ].copy()
    part["model"] = part["models"].astype(str)

    for x, model in enumerate(model_order):
        sub = part[part["model"] == model].set_index("representation")
        if not {"nn1", "classifier"}.issubset(sub.index):
            continue
        y_np = float(sub.loc["nn1", "spearman_rho"])
        y_cl = float(sub.loc["classifier", "spearman_rho"])
        ax.plot([x, x], [y_cl, y_np], color="0.72", lw=0.8, zorder=1)
        ax.scatter(
            [x], [y_np], s=28, c=COLORS["np"], marker="o",
            edgecolors="white", linewidths=0.4, zorder=3,
            label="NP" if x == 0 else None,
        )
        ax.scatter(
            [x], [y_cl], s=28, c=COLORS["classifier"], marker="s",
            edgecolors="white", linewidths=0.4, zorder=3,
            label="Classifier" if x == 0 else None,
        )

    ax.axvspan(-0.5, 4.5, color="0.95", zorder=0)
    ax.axvline(4.5, color="0.75", lw=0.6, linestyle="--")
    ax.text(
        2.0, 0.965, "used in training",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=6.4, color="0.38",
    )
    ax.text(
        6.0, 0.965, "not used in training",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=6.4, color="0.38",
    )
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(
        [figure_model_label(x) for x in model_order],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel(r"Spearman $\rho(D_z,D_{\mathrm{behavior}})$")
    ax.set_title(TRIAL_LABELS[trial])
    ax.legend(frameon=False, loc="lower right")


def per_model_np_classifier_delta(geometry: pd.DataFrame) -> pd.DataFrame:
    keep_models = ["m4", "m5", "m6", "m7", "m8", "m1", "m3", "m9"]
    part = geometry[
        (geometry["protocol"] == "generalization_m4_m8")
        & (geometry["scope"] == "within_model")
        & (geometry["behavior_distance"] == PRIMARY_BEHAVIOR_DISTANCE)
        & (geometry["representation"].isin(["nn1", "classifier"]))
    ].copy()
    part["model"] = part["models"].astype(str)
    part = part[part["model"].isin(keep_models)]

    rows = []
    for (trial, model), group in part.groupby(["trial_data_name", "model"]):
        vals = group.set_index("representation")
        if not {"nn1", "classifier"}.issubset(vals.index):
            continue
        rho_np = float(vals.loc["nn1", "spearman_rho"])
        rho_cl = float(vals.loc["classifier", "spearman_rho"])
        rows.append(
            {
                "trial_data_name": trial,
                "model": model,
                "paper_label": figure_model_label(model),
                "rho_np": rho_np,
                "rho_classifier": rho_cl,
                "delta_rho": rho_np - rho_cl,
                "status": "training" if model in TRAIN_MODELS else "heldout",
            }
        )
    return pd.DataFrame(rows)


def draw_per_model_delta_geometry(ax, delta: pd.DataFrame):
    model_order = ["m4", "m5", "m6", "m7", "m8", "m1", "m3", "m9"]
    x = np.arange(len(model_order), dtype=float)
    offsets = {
        "pub_Konstantinidis2020": -0.08,
        "pub_Thrailkill2023": 0.08,
    }
    colors = {
        "pub_Konstantinidis2020": COLORS["design_380"],
        "pub_Thrailkill2023": COLORS["design_27"],
    }
    markers = {
        "pub_Konstantinidis2020": "o",
        "pub_Thrailkill2023": "s",
    }

    for trial in TRIAL_DATA_NAMES:
        sub = delta[delta["trial_data_name"] == trial].set_index("model")
        xx, yy = [], []
        for i, model in enumerate(model_order):
            if model not in sub.index:
                continue
            xx.append(x[i] + offsets[trial])
            yy.append(float(sub.loc[model, "delta_rho"]))
        ax.scatter(
            xx,
            yy,
            s=25,
            marker=markers[trial],
            color=colors[trial],
            edgecolors="white",
            linewidths=0.35,
            label=TRIAL_SHORT[trial],
            zorder=3,
        )

    ax.axhline(0, color="0.45", lw=0.7, linestyle="--")
    ax.axvspan(-0.5, 4.5, color="0.95", zorder=0)
    ax.axvline(4.5, color="0.75", lw=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [figure_model_label(m) for m in model_order],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel(r"$\Delta\rho=\rho_{\mathrm{NP}}-\rho_{\mathrm{Classifier}}$")
    ax.set_title("NP − classifier within-model fidelity")
    ax.legend(frameon=False, loc="best")


def make_figure_03_geometry_fidelity():
    geometry = load_geometry_table()

    source_geom = geometry[
        (geometry["protocol"] == "generalization_m4_m8")
        & (geometry["behavior_distance"] == PRIMARY_BEHAVIOR_DISTANCE)
        & (geometry["representation"].isin(["nn1", "classifier"]))
        & (geometry["scope"] == "within_model")
        & (geometry["models"].astype(str).isin(
            ["m4", "m5", "m6", "m7", "m8", "m1", "m3", "m9"]
        ))
    ].copy()
    delta = per_model_np_classifier_delta(geometry)
    write_source(source_geom, "figure_03_within_model_geometry.csv")
    write_source(delta, "figure_03_np_minus_classifier.csv")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.65),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.15]},
    )
    draw_within_model_geometry(
        axes[0], geometry, "pub_Konstantinidis2020"
    )
    draw_within_model_geometry(
        axes[1], geometry, "pub_Thrailkill2023"
    )
    draw_per_model_delta_geometry(axes[2], delta)
    for letter, ax in zip("ABC", axes, strict=True):
        annotate_panel(ax, letter)
    fig.subplots_adjust(wspace=0.38, bottom=0.25)
    print_figure_03_results(delta)
    return fig


def support_family_dir(support_root: Path, trial: str) -> Path:
    return (
        support_root
        / "primary"
        / trial
        / "parallel_m4m8_encoder"
        / "training_family_support"
    )


def load_cmax_subjects(support_root: Path, trial: str) -> pd.DataFrame:
    root = support_family_dir(support_root, trial)
    sim = require_csv(root / "training_simulation_subjects.csv")
    app = require_csv(root / "application_subjects.csv")
    sim = sim[
        (sim["representation"] == "nn1")
        & (sim["score"] == PRIMARY_SCORE)
    ].copy()
    sim["plot_group"] = "Training simulations"
    app = app[
        (app["representation"] == "nn1")
        & (app["score"] == PRIMARY_SCORE)
        & (app["query_data_name"].isin(["r1", "m9"]))
    ].copy()
    app["plot_group"] = app["query_data_name"].map(
        {"m9": "Held-out H1", "r1": "Real data"}
    )
    out = pd.concat([sim, app], ignore_index=True)
    out["trial_data_name"] = trial
    return out


def draw_cmax_panel(ax, frame: pd.DataFrame, trial: str):
    groups = ["Training simulations", "Held-out H1", "Real data"]
    arrays = [
        frame.loc[frame["plot_group"] == g, "c_max"].dropna().to_numpy()
        for g in groups
    ]
    positions = np.arange(1, 4)
    vp = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.76,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    group_colors = [
        COLORS["training_sim"],
        COLORS["heuristic"],
        COLORS["real"],
    ]
    for body, color in zip(vp["bodies"], group_colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    vp["cmedians"].set_color(group_colors)
    vp["cmedians"].set_linewidth(1.2)


    for pos, values, color in zip(
        positions, arrays, group_colors, strict=True
    ):
        xx, yy = jitter_points(
            values,
            n_max=160 if pos != 3 else 300,
            seed=SEED + pos,
            width=0.075,
        )
        ax.scatter(
            pos + xx,
            yy,
            s=5.0,
            c=color,
            alpha=0.25 if pos != 3 else 0.48,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

    ax.axhline(
        0.05,
        color="0.35",
        lw=0.7,
        linestyle="--",
        label=r"$C_{\max}=.05$",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(["Train sims", "H1", "Real"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(r"Maximum family compatibility $C_{\max}$")
    ax.set_title(TRIAL_LABELS[trial])
    ax.legend(frameon=False, loc="upper right")


def load_auroc_bootstrap(support_root: Path) -> pd.DataFrame:
    path = (
        support_root
        / "aggregate"
        / "primary_m4m8_training_family_bootstrap_95ci.csv"
    )
    boot = require_csv(path)
    return boot[
        (boot["score"] == PRIMARY_SCORE)
        & (boot["statistic"] == "real_vs_simulation_AUROC")
        & (boot["representation"].isin(MAIN_REPRESENTATIONS))
        & (boot["application_query"].isin(["r1", "m9"]))
    ].copy()


def draw_auroc_panel(ax, boot: pd.DataFrame, query: str, title: str):
    reps = [
        rep for rep in MAIN_REPRESENTATIONS
        if rep in set(boot["representation"])
    ]
    x = np.arange(len(reps), dtype=float)
    offsets = {
        "pub_Konstantinidis2020": -0.10,
        "pub_Thrailkill2023": 0.10,
    }
    colors = {
        "pub_Konstantinidis2020": COLORS["design_380"],
        "pub_Thrailkill2023": COLORS["design_27"],
    }
    markers = {
        "pub_Konstantinidis2020": "o",
        "pub_Thrailkill2023": "s",
    }

    for trial in TRIAL_DATA_NAMES:
        ys, lows, highs = [], [], []
        valid_x = []
        for i, rep in enumerate(reps):
            row = boot[
                (boot["trial_data_name"] == trial)
                & (boot["application_query"] == query)
                & (boot["representation"] == rep)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            valid_x.append(x[i] + offsets[trial])
            ys.append(float(row["estimate"]))
            lows.append(float(row["ci_low"]))
            highs.append(float(row["ci_high"]))
        if not ys:
            continue
        ys = np.asarray(ys)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        ax.errorbar(
            valid_x,
            ys,
            yerr=[ys - lows, highs - ys],
            fmt=markers[trial],
            color=colors[trial],
            ecolor=colors[trial],
            capsize=2,
            markersize=4,
            elinewidth=0.9,
            label=TRIAL_SHORT[trial],
        )

    ax.axhline(0.5, color="0.55", lw=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([REP_LABEL[r] for r in reps], rotation=35, ha="right")
    ax.set_ylim(0.45, 1.015)
    ax.set_ylabel(r"AUROC using anomaly score $1-C_{\max}$")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")


def make_figure_04_mismatch(support_root: Path):
    cmax = pd.concat(
        [load_cmax_subjects(support_root, t) for t in TRIAL_DATA_NAMES],
        ignore_index=True,
    )
    boot = load_auroc_bootstrap(support_root)
    write_source(cmax, "figure_04_cmax_subjects.csv")
    write_source(boot, "figure_04_auroc_bootstrap.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    draw_cmax_panel(
        axes[0, 0],
        cmax[cmax["trial_data_name"] == "pub_Konstantinidis2020"],
        "pub_Konstantinidis2020",
    )
    draw_cmax_panel(
        axes[0, 1],
        cmax[cmax["trial_data_name"] == "pub_Thrailkill2023"],
        "pub_Thrailkill2023",
    )
    draw_auroc_panel(
        axes[1, 0],
        boot,
        "r1",
        "Real data vs training-family simulations",
    )
    draw_auroc_panel(
        axes[1, 1],
        boot,
        "m9",
        "Held-out H1 vs training-family simulations",
    )
    for letter, ax in zip("ABCD", axes.flat, strict=True):
        annotate_panel(ax, letter)
    fig.subplots_adjust(hspace=0.43, wspace=0.31)
    print_figure_04_results(cmax, boot)
    return fig


def print_main_figure_03_mismatch_results(cmax: pd.DataFrame):
    print_figure_header("MAIN Figure 3 numeric summary")
    print("NP training-family compatibility C_max")
    groups = ["Training simulations", "Held-out H1", "Real data"]
    for trial in TRIAL_DATA_NAMES:
        print(f"{TRIAL_LABELS[trial]}")
        part = cmax[cmax["trial_data_name"] == trial]
        for group in groups:
            values = part.loc[part["plot_group"] == group, "c_max"].dropna()
            if values.empty:
                continue
            print(
                f"  {group:<22}  "
                f"median={_fmt(values.median())}  "
                f"P(C_max<0.05)={_fmt((values < 0.05).mean())}"
            )


def make_main_figure_03_mismatch(support_root: Path):
    cmax = pd.concat(
        [load_cmax_subjects(support_root, t) for t in TRIAL_DATA_NAMES],
        ignore_index=True,
    )
    write_source(cmax, "main_figure_03_cmax_subjects.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), sharey=True)
    for letter, trial, ax in zip("AB", TRIAL_DATA_NAMES, axes, strict=True):
        draw_cmax_panel(
            ax,
            cmax[cmax["trial_data_name"] == trial],
            trial,
        )
        annotate_panel(ax, letter)

    axes[1].set_ylabel("")
    fig.subplots_adjust(wspace=0.18, bottom=0.18)
    print_main_figure_03_mismatch_results(cmax)
    return fig


def make_figure_s1_all_geometry():
    geometry = load_geometry_table()
    keep_models = ["m4", "m5", "m6", "m7", "m8", "m1", "m3"]
    part = geometry[
        (geometry["protocol"] == "generalization_m4_m8")
        & (geometry["behavior_distance"] == PRIMARY_BEHAVIOR_DISTANCE)
        & (geometry["scope"] == "within_model")
        & (geometry["models"].astype(str).isin(keep_models))
        & (geometry["representation"].isin(ALL_REPRESENTATIONS))
    ].copy()
    part["model"] = part["models"].astype(str)
    part["group"] = np.where(
        part["model"].isin(TRAIN_MODELS),
        "Training simulators",
        "Held-out OW1/AW1",
    )
    summary = (
        part.groupby(["trial_data_name", "representation", "group"], as_index=False)
        ["spearman_rho"]
        .mean()
        .rename(columns={"spearman_rho": "mean_within_model_rho"})
    )
    write_source(part, "figure_s1_all_representation_geometry_within_model.csv")
    write_source(summary, "figure_s1_all_representation_geometry_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for letter, trial, ax in zip("AB", TRIAL_DATA_NAMES, axes, strict=True):
        sub = summary[summary["trial_data_name"] == trial]
        reps = [
            rep for rep in ALL_REPRESENTATIONS
            if rep in set(sub["representation"])
        ]
        x = np.arange(len(reps), dtype=float)
        for group, marker, offset, label in [
            ("Training simulators", "o", -0.08, "Training OW2–OW5, AW2"),
            ("Held-out OW1/AW1", "s", 0.08, "Held-out OW1/AW1"),
        ]:
            vals = sub[sub["group"] == group].set_index("representation")
            xs, ys = [], []
            for i, rep in enumerate(reps):
                if rep not in vals.index:
                    continue
                xs.append(x[i] + offset)
                ys.append(float(vals.loc[rep, "mean_within_model_rho"]))
            ax.plot(
                xs,
                ys,
                marker=marker,
                linestyle="none",
                markersize=4.2,
                color="0.35" if group == "Held-out OW1/AW1" else COLORS["np"],
                label=label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([REP_LABEL[r] for r in reps], rotation=35, ha="right")
        ax.set_title(TRIAL_LABELS[trial])
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, loc="lower left")
        annotate_panel(ax, letter)
    axes[0].set_ylabel(r"Mean within-model Spearman $\rho$")
    fig.subplots_adjust(bottom=0.28, wspace=0.12)
    print_figure_s1_results(summary)
    return fig


def make_figure_s2_family_generalization(support_root: Path):
    path = (
        support_root
        / "aggregate"
        / "primary_m4m8_training_family_known_recovery_summary.csv"
    )
    frame = require_csv(path)
    frame = frame[
        (frame["score"] == PRIMARY_SCORE)
        & (frame["representation"].isin(MAIN_REPRESENTATIONS))
    ].copy()
    write_source(frame, "figure_s2_heldout_family_recovery.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95), sharey=True)
    model_cols = [("accuracy_m1", "OW1"), ("accuracy_m3", "AW1")]
    for letter, trial, ax in zip("AB", TRIAL_DATA_NAMES, axes, strict=True):
        sub = frame[frame["trial_data_name"] == trial]
        reps = [
            rep for rep in MAIN_REPRESENTATIONS
            if rep in set(sub["representation"])
        ]
        x = np.arange(len(model_cols))
        offsets = np.linspace(-0.18, 0.18, max(len(reps), 1))
        for off, rep in zip(offsets, reps, strict=True):
            row = sub[sub["representation"] == rep]
            if row.empty:
                continue
            row = row.iloc[0]
            y = [float(row[col]) for col, _ in model_cols]
            ax.plot(
                x + off,
                y,
                marker="o",
                linestyle="-",
                linewidth=0.7,
                markersize=3.8,
                color=REP_COLORS[rep],
                label=REP_LABEL[rep],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in model_cols])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Correct coarse-family assignment")
        ax.set_title(TRIAL_LABELS[trial])
        ax.legend(frameon=False, fontsize=6.5)
        annotate_panel(ax, letter)
    fig.subplots_adjust(wspace=0.15)
    print_figure_s2_results(frame)
    return fig


def make_figure_s3_score_robustness(support_root: Path):
    path = (
        support_root
        / "aggregate"
        / "primary_m4m8_training_family_real_vs_simulation_summary.csv"
    )
    frame = require_csv(path)
    frame = frame[
        (frame["representation"] == "nn1")
        & (frame["application_query"].isin(["r1", "m9"]))
        & (frame["score"].isin(["mean_euclidean", "knn", "gaussian_nll"]))
    ].copy()
    write_source(frame, "figure_s3_support_metric_robustness.csv")

    score_order = ["mean_euclidean", "knn", "gaussian_nll"]
    score_labels = ["Mean Euclidean", "kNN", "Gaussian NLL"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.85), sharey=True)
    for letter, query, title, ax in [
        ("A", "r1", "Real data", axes[0]),
        ("B", "m9", "Held-out H1", axes[1]),
    ]:
        x = np.arange(len(score_order))
        for trial, marker, color in [
            ("pub_Konstantinidis2020", "o", COLORS["design_380"]),
            ("pub_Thrailkill2023", "s", COLORS["design_27"]),
        ]:
            sub = frame[
                (frame["trial_data_name"] == trial)
                & (frame["application_query"] == query)
            ].set_index("score")
            y = [
                float(sub.loc[s, "auroc_real_vs_simulation"])
                if s in sub.index
                else np.nan
                for s in score_order
            ]
            ax.plot(
                x,
                y,
                marker=marker,
                linewidth=1.0,
                markersize=4.5,
                color=color,
                label=TRIAL_SHORT[trial],
            )
        ax.axhline(0.5, color="0.6", lw=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(score_labels, rotation=20, ha="right")
        ax.set_ylim(0.30, 1.02)
        ax.set_ylabel("Application-vs-simulation AUROC")
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower left")
        annotate_panel(ax, letter)
    fig.subplots_adjust(bottom=0.24, wspace=0.14)
    print_figure_s3_results(frame)
    return fig


def make_figure_s4_interpolation():
    path = geometry_root() / "np_generative_interpolation_summary.csv"
    frame = optional_csv(path)
    if frame.empty:
        return None
    frame = frame[
        (frame["protocol"] == "generalization_m4_m8")
        & (frame["representation"] == "nn1")
    ].copy()
    if frame.empty:
        return None
    write_source(frame, "figure_s4_np_interpolation_summary.csv")

    pair_order = [
        "within_option_m4",
        "training_cross_family_m4_m8",
        "heldout_known_m1_m3",
        "heldout_to_heuristic_m1_m9",
    ]
    pair_labels = {
        "within_option_m4": "OW2→OW2",
        "training_cross_family_m4_m8": "OW2→AW2",
        "heldout_known_m1_m3": "OW1→AW1",
        "heldout_to_heuristic_m1_m9": "OW1→H1",
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharey=True)
    for letter, trial, ax in zip("AB", TRIAL_DATA_NAMES, axes, strict=True):
        sub = frame[frame["trial_data_name"] == trial].set_index("pair")
        pairs = [p for p in pair_order if p in sub.index]
        x = np.arange(len(pairs))
        true_y = [float(sub.loc[p, "true_endpoint_distance"]) for p in pairs]
        dec_y = [float(sub.loc[p, "decoded_endpoint_distance"]) for p in pairs]
        for i, (a, b) in enumerate(zip(true_y, dec_y, strict=True)):
            ax.plot([i, i], [a, b], color="0.75", lw=0.8)
        ax.scatter(x, true_y, s=28, marker="o", color="0.25", label="True behavioral distance")
        ax.scatter(x, dec_y, s=28, marker="s", color=COLORS["np"], label="NP-decoded endpoint distance")
        ax.set_xticks(x)
        ax.set_xticklabels([pair_labels[p] for p in pairs])
        ax.set_ylabel("Behavioral-function RMSE")
        ax.set_title(TRIAL_LABELS[trial])
        ax.legend(frameon=False, fontsize=6.7, loc="upper left")
        annotate_panel(ax, letter)
    fig.subplots_adjust(wspace=0.16)
    print_figure_s4_results(frame)
    return fig


def compute_training_model_recovery_for_trial(trial_data_name: str):
    core, _ = load_core_config()


    old_trial = getattr(core, "TRIAL_DATA_NAME", None)
    old_k = getattr(core, "KNN_K", None)
    old_scores = getattr(core, "SCORE_KINDS", None)
    old_primary = getattr(core, "PRIMARY_SCORE", None)
    old_reps = getattr(core, "REPRESENTATION_KINDS", None)

    try:
        core.TRIAL_DATA_NAME = trial_data_name
        core.KNN_K = int(MODEL_RECOVERY_KNN_K)
        core.SCORE_KINDS = (PRIMARY_SCORE,)
        core.PRIMARY_SCORE = PRIMARY_SCORE
        core.REPRESENTATION_KINDS = tuple(MAIN_REPRESENTATIONS)

        experiment = core.load_experiment_data(
            trial_data_name,
            data_names=TRAIN_MODELS,
            seed=SEED,
            full_real=False,
        )
        rep_outputs = core.encode_protocol_representations(
            experiment,
            TRAIN_MODELS,
        )

        score_tables = []
        for rep_label, info in rep_outputs.items():
            if rep_label not in MAIN_REPRESENTATIONS:
                continue
            supports, _ = core.build_supports(
                info["latent"],
                experiment,
                TRAIN_MODELS,
            )
            for query_name in TRAIN_MODELS:
                query_idx = core.indices_for_data_split(
                    experiment,
                    query_name,
                    "test",
                )
                table = core.score_query_set(
                    info["latent"],
                    experiment,
                    supports,
                    query_name,
                    query_idx,
                    rep_label,
                    "training_model_recovery",
                )
                if not table.empty:
                    score_tables.append(table)

        scores = (
            pd.concat(score_tables, ignore_index=True)
            if score_tables
            else pd.DataFrame()
        )
        best = core.best_from_long(scores)
        if best.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        best = best[best["score"] == PRIMARY_SCORE].copy()
        best["correct_exact"] = (
            best["best_model"].astype(str)
            == best["query_true_model"].astype(str)
        )
        best["trial_data_name"] = trial_data_name

        summary_rows = []
        confusion_rows = []
        for rep in MAIN_REPRESENTATIONS:
            sub = best[best["representation"] == rep]
            if sub.empty:
                continue

            summary_rows.append(
                {
                    "trial_data_name": trial_data_name,
                    "representation": rep,
                    "true_model": "__overall__",
                    "n_subjects": int(len(sub)),
                    "exact_model_accuracy": float(
                        sub["correct_exact"].astype(float).mean()
                    ),
                }
            )
            for model in TRAIN_MODELS:
                model_sub = sub[sub["query_true_model"] == model]
                if model_sub.empty:
                    continue
                summary_rows.append(
                    {
                        "trial_data_name": trial_data_name,
                        "representation": rep,
                        "true_model": model,
                        "n_subjects": int(len(model_sub)),
                        "exact_model_accuracy": float(
                            model_sub["correct_exact"].astype(float).mean()
                        ),
                    }
                )

                counts = (
                    model_sub["best_model"]
                    .astype(str)
                    .value_counts()
                    .reindex(TRAIN_MODELS, fill_value=0)
                )
                denom = max(int(counts.sum()), 1)
                for predicted_model, count in counts.items():
                    confusion_rows.append(
                        {
                            "trial_data_name": trial_data_name,
                            "representation": rep,
                            "true_model": model,
                            "predicted_model": predicted_model,
                            "count": int(count),
                            "proportion": float(count / denom),
                        }
                    )

        return (
            best,
            pd.DataFrame(summary_rows),
            pd.DataFrame(confusion_rows),
        )
    finally:
        if old_trial is not None:
            core.TRIAL_DATA_NAME = old_trial
        if old_k is not None:
            core.KNN_K = old_k
        if old_scores is not None:
            core.SCORE_KINDS = old_scores
        if old_primary is not None:
            core.PRIMARY_SCORE = old_primary
        if old_reps is not None:
            core.REPRESENTATION_KINDS = old_reps


def load_training_model_recovery():
    subject_parts = []
    summary_parts = []
    confusion_parts = []
    for trial in TRIAL_DATA_NAMES:
        subjects, summary, confusion = compute_training_model_recovery_for_trial(
            trial
        )
        if not subjects.empty:
            subject_parts.append(subjects)
        if not summary.empty:
            summary_parts.append(summary)
        if not confusion.empty:
            confusion_parts.append(confusion)

    subjects = (
        pd.concat(subject_parts, ignore_index=True)
        if subject_parts
        else pd.DataFrame()
    )
    summary = (
        pd.concat(summary_parts, ignore_index=True)
        if summary_parts
        else pd.DataFrame()
    )
    confusion = (
        pd.concat(confusion_parts, ignore_index=True)
        if confusion_parts
        else pd.DataFrame()
    )
    return subjects, summary, confusion


def draw_recovery_overall_panel(ax, summary: pd.DataFrame):
    reps = [
        rep for rep in MAIN_REPRESENTATIONS
        if rep in set(summary["representation"])
    ]
    x = np.arange(len(reps), dtype=float)
    offsets = {
        "pub_Konstantinidis2020": -0.10,
        "pub_Thrailkill2023": 0.10,
    }
    colors = {
        "pub_Konstantinidis2020": COLORS["design_380"],
        "pub_Thrailkill2023": COLORS["design_27"],
    }
    markers = {
        "pub_Konstantinidis2020": "o",
        "pub_Thrailkill2023": "s",
    }
    for trial in TRIAL_DATA_NAMES:
        sub = summary[
            (summary["trial_data_name"] == trial)
            & (summary["true_model"] == "__overall__")
        ].set_index("representation")
        xx, yy = [], []
        for i, rep in enumerate(reps):
            if rep not in sub.index:
                continue
            xx.append(x[i] + offsets[trial])
            yy.append(float(sub.loc[rep, "exact_model_accuracy"]))
        ax.scatter(
            xx,
            yy,
            s=28,
            marker=markers[trial],
            color=colors[trial],
            edgecolors="white",
            linewidths=0.35,
            label=TRIAL_SHORT[trial],
            zorder=3,
        )
    ax.axhline(
        1 / len(TRAIN_MODELS),
        color="0.55",
        lw=0.7,
        linestyle="--",
        label="Chance (1/5)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([REP_LABEL[r] for r in reps], rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Exact model-recovery accuracy")
    ax.set_title("Overall five-way recovery")
    ax.legend(frameon=False, fontsize=6.7, loc="best")


def draw_recovery_per_model_panel(
    ax,
    summary: pd.DataFrame,
    trial_data_name: str,
):
    sub = summary[
        (summary["trial_data_name"] == trial_data_name)
        & (summary["true_model"].isin(TRAIN_MODELS))
    ].copy()
    reps = [
        rep for rep in MAIN_REPRESENTATIONS
        if rep in set(sub["representation"])
    ]
    x = np.arange(len(TRAIN_MODELS), dtype=float)
    offsets = np.linspace(-0.18, 0.18, max(len(reps), 1))

    for offset, rep in zip(offsets, reps, strict=True):
        vals = sub[sub["representation"] == rep].set_index("true_model")
        yy = [
            float(vals.loc[model, "exact_model_accuracy"])
            if model in vals.index
            else np.nan
            for model in TRAIN_MODELS
        ]
        ax.plot(
            x + offset,
            yy,
            marker="o",
            linestyle="none",
            markersize=4.2,
            color=REP_COLORS[rep],
            label=REP_LABEL[rep],
        )

    ax.axhline(
        1 / len(TRAIN_MODELS),
        color="0.6",
        lw=0.7,
        linestyle="--",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([figure_model_label(m) for m in TRAIN_MODELS])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Exact model-recovery accuracy")
    ax.set_title(TRIAL_LABELS[trial_data_name])
    ax.legend(frameon=False, fontsize=6.2, loc="best")


def make_figure_s5_training_model_recovery():
    subjects, summary, confusion = load_training_model_recovery()
    if summary.empty:
        warnings.warn(
            "Figure S5 model recovery could not be generated: no recovery "
            f"results were produced from the current {CFG_NAME} checkpoints."
        )
        return None

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.35, 2.85),
        gridspec_kw={"width_ratios": [1.05, 1.15, 1.15]},
    )
    draw_recovery_overall_panel(axes[0], summary)
    draw_recovery_per_model_panel(
        axes[1],
        summary,
        "pub_Konstantinidis2020",
    )
    draw_recovery_per_model_panel(
        axes[2],
        summary,
        "pub_Thrailkill2023",
    )
    for letter, ax in zip("ABC", axes, strict=True):
        annotate_panel(ax, letter)
    fig.subplots_adjust(bottom=0.25, wspace=0.36)

    print_figure_s5_results(summary, confusion)
    return fig


def _load_support_metric_robustness_frame(support_root: Path) -> pd.DataFrame:
    path = (
        support_root
        / "aggregate"
        / "primary_m4m8_training_family_real_vs_simulation_summary.csv"
    )
    frame = require_csv(path)
    return frame[
        (frame["representation"] == "nn1")
        & (frame["application_query"].isin(["r1", "m9"]))
        & (frame["score"].isin(["mean_euclidean", "knn", "gaussian_nll"]))
    ].copy()


def _draw_metric_robustness_panel(
    ax,
    frame: pd.DataFrame,
    query: str,
    title: str,
):
    score_order = ["mean_euclidean", "knn", "gaussian_nll"]
    score_labels = ["Mean Euclidean", "kNN", "Gaussian NLL"]
    x = np.arange(len(score_order))
    for trial, marker, color in [
        ("pub_Konstantinidis2020", "o", COLORS["design_380"]),
        ("pub_Thrailkill2023", "s", COLORS["design_27"]),
    ]:
        sub = frame[
            (frame["trial_data_name"] == trial)
            & (frame["application_query"] == query)
        ].set_index("score")
        y = [
            float(sub.loc[s, "auroc_real_vs_simulation"])
            if s in sub.index
            else np.nan
            for s in score_order
        ]
        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=1.0,
            markersize=4.5,
            color=color,
            label=TRIAL_SHORT[trial],
        )
    ax.axhline(0.5, color="0.6", lw=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(score_labels, rotation=18, ha="right")
    ax.set_ylim(0.30, 1.02)
    ax.set_ylabel("Application-vs-simulation AUROC")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")


def make_appendix_figure_s4_diagnostic_robustness(support_root: Path):
    boot = load_auroc_bootstrap(support_root)
    metric = _load_support_metric_robustness_frame(support_root)

    write_source(boot, "appendix_s4_representation_auroc.csv")
    write_source(metric, "appendix_s4_support_metric_robustness.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05))

    draw_auroc_panel(
        axes[0, 0],
        boot,
        "r1",
        "Real data: representation objective",
    )
    draw_auroc_panel(
        axes[0, 1],
        boot,
        "m9",
        "H1: representation objective",
    )
    _draw_metric_robustness_panel(
        axes[1, 0],
        metric,
        "r1",
        "Real data: support estimator",
    )
    _draw_metric_robustness_panel(
        axes[1, 1],
        metric,
        "m9",
        "H1: support estimator",
    )

    for letter, ax in zip("ABCD", axes.flat, strict=True):
        annotate_panel(ax, letter)
    fig.subplots_adjust(hspace=0.46, wspace=0.30)

    print_figure_header("Appendix Figure S4 numeric summary")
    print("Top row: AUROC across representation objectives.")
    print("Bottom row: NP AUROC across support estimators.")
    print_figure_04_results(
        pd.concat(
            [load_cmax_subjects(support_root, t) for t in TRIAL_DATA_NAMES],
            ignore_index=True,
        ),
        boot,
    )
    print_figure_s3_results(metric)
    return fig


@torch.no_grad()
def infer_np_posterior_parameters(
    core,
    rep,
    experiment,
    indices,
):
    data = replace(
        experiment.choice_data,
        trials=core.apply_checkpoint_standardizer(
            experiment.choice_data.trials,
            rep.checkpoint.get("standardizer"),
        ),
    )

    indices = torch.as_tensor(
        np.asarray(indices, dtype=np.int64),
        dtype=torch.long,
    )
    if len(indices) == 0:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    mean_parts = []
    logvar_parts = []
    n_trial_parts = []

    for start in range(0, len(indices), core.INFERENCE_BATCH_SIZE):
        batch_indices = indices[
            start : start + core.INFERENCE_BATCH_SIZE
        ]

        trials = core.batch_trials(
            data,
            batch_indices,
        ).to(core.DEVICE)
        choices = data.choices[
            batch_indices
        ].to(core.DEVICE)
        mask = data.trial_mask[
            batch_indices
        ].to(core.DEVICE)

        mean, logvar = rep.model.encode_context(
            trials,
            choices,
            mask,
        )

        mean_parts.append(mean.detach().cpu())
        logvar_parts.append(logvar.detach().cpu())


        n_valid = mask.reshape(mask.shape[0], -1).sum(dim=1)
        n_trial_parts.append(n_valid.detach().cpu())

    means = torch.cat(mean_parts, dim=0).numpy().astype(np.float64)
    logvars = torch.cat(logvar_parts, dim=0).numpy().astype(np.float64)
    n_trials = (
        torch.cat(n_trial_parts, dim=0)
        .numpy()
        .astype(np.int64)
    )
    return means, logvars, n_trials


def build_posterior_diagnostic_frames():
    core, _ = load_core_config()
    rep = core.load_representation(
        POSTERIOR_REPRESENTATION,
        TRAIN_MODELS,
    )
    if rep is None:
        raise FileNotFoundError(
            f"{CFG_NAME} NP checkpoint for general_400 / m4--m8 was not found."
        )
    np_kinds = core.NP_LATENT_KINDS | core.NP_CLASSIFIER_KINDS
    if getattr(rep, "kind", None) not in np_kinds:
        raise TypeError(
            "Posterior-variance diagnostics require an NP representation "
            "with encode_context returning (mean, logvar)."
        )

    subject_rows = []
    dimension_rows = []
    summary_rows = []

    for trial in POSTERIOR_DESIGNS:
        experiment = core.load_experiment_data(
            trial,
            data_names=POSTERIOR_DIAGNOSTIC_MODELS,
            seed=SEED,
            full_real=False,
        )

        all_means = []
        all_logvars = []
        all_n_trials = []

        for model in POSTERIOR_DIAGNOSTIC_MODELS:
            indices = test_indices_for_group(experiment, model)
            if len(indices) == 0:
                continue

            means, logvars, n_trials = infer_np_posterior_parameters(
                core,
                rep,
                experiment,
                indices,
            )
            if len(means) == 0:
                continue

            all_means.append(means)
            all_logvars.append(logvars)
            all_n_trials.append(n_trials)


            for i in range(len(means)):
                subject_rows.append(
                    {
                        "trial_data_name": trial,
                        "model": model,
                        "paper_label": figure_model_label(model),
                        "combined_subject_index": int(indices[i]),
                        "n_trials": int(n_trials[i]),
                        "subject_mean_log_variance": float(
                            np.mean(logvars[i])
                        ),
                        "subject_mean_posterior_variance": float(
                            np.mean(np.exp(logvars[i]))
                        ),
                        "subject_mean_posterior_sd": float(
                            np.mean(np.exp(0.5 * logvars[i]))
                        ),
                    }
                )


            model_mu_var = np.var(means, axis=0, ddof=1)
            model_mean_logvar = np.mean(logvars, axis=0)
            for dim in range(means.shape[1]):
                dimension_rows.append(
                    {
                        "trial_data_name": trial,
                        "model": model,
                        "paper_label": figure_model_label(model),
                        "latent_dimension": dim + 1,
                        "mu_variance": float(model_mu_var[dim]),
                        "mean_log_variance": float(
                            model_mean_logvar[dim]
                        ),
                    }
                )

        if not all_means:
            continue

        means = np.vstack(all_means)
        logvars = np.vstack(all_logvars)
        n_trials = np.concatenate(all_n_trials)

        mu_var_by_dim = np.var(means, axis=0, ddof=1)
        mean_logvar_by_dim = np.mean(logvars, axis=0)


        for dim in range(means.shape[1]):
            dimension_rows.append(
                {
                    "trial_data_name": trial,
                    "model": "__all_retained_models__",
                    "paper_label": "All retained OW/AW",
                    "latent_dimension": dim + 1,
                    "mu_variance": float(mu_var_by_dim[dim]),
                    "mean_log_variance": float(
                        mean_logvar_by_dim[dim]
                    ),
                }
            )

        summary_rows.append(
            {
                "trial_data_name": trial,
                "design_label": POSTERIOR_DESIGN_LABELS[trial],
                "n_subjects": int(len(means)),
                "mean_valid_trials": float(np.mean(n_trials)),
                "median_valid_trials": float(np.median(n_trials)),
                "min_valid_trials": int(np.min(n_trials)),
                "max_valid_trials": int(np.max(n_trials)),

                "mean_mu_variance": float(
                    np.mean(mu_var_by_dim)
                ),
                "sum_mu_variance": float(
                    np.sum(mu_var_by_dim)
                ),

                "mean_log_variance": float(
                    np.mean(logvars)
                ),
                "mean_posterior_variance": float(
                    np.mean(np.exp(logvars))
                ),
                "mean_posterior_sd": float(
                    np.mean(np.exp(0.5 * logvars))
                ),
            }
        )

    subjects = pd.DataFrame(subject_rows)
    dimensions = pd.DataFrame(dimension_rows)
    summary = pd.DataFrame(summary_rows)

    write_source(
        subjects,
        "figure_s6_np_posterior_subjects.csv",
    )
    write_source(
        dimensions,
        "figure_s6_np_posterior_dimensions.csv",
    )
    write_source(
        summary,
        "figure_s6_np_posterior_summary.csv",
    )

    return subjects, dimensions, summary


def print_posterior_diagnostic_results(
    subjects: pd.DataFrame,
    summary: pd.DataFrame,
):
    print_figure_header("Appendix Figure S6 numeric summary")
    print(
        "NP posterior diagnostics use ALL test subjects from the same retained "
        "OW/AW model set in every dataset: OW1-OW5 and AW1-AW2."
    )
    print(
        "mean Var_s(mu_d) measures between-subject latent spread; "
        "mean log variance measures q(z|D) uncertainty "
        "(lower / more negative = more certain)."
    )

    indexed = summary.set_index("trial_data_name")
    for trial in POSTERIOR_PLOT_ORDER:
        if trial not in indexed.index:
            continue
        row = indexed.loc[trial]
        print("")
        print(POSTERIOR_DESIGN_LABELS[trial])
        print(
            f"  n test subjects = {int(row['n_subjects'])}"
        )
        print(
            "  valid trials: "
            f"mean={_fmt(row['mean_valid_trials'], 1)}, "
            f"median={_fmt(row['median_valid_trials'], 1)}, "
            f"range=[{int(row['min_valid_trials'])}, "
            f"{int(row['max_valid_trials'])}]"
        )
        print(
            f"  mean Var_s(mu_d) = "
            f"{_fmt(row['mean_mu_variance'], 6)}"
        )
        print(
            f"  mean log Var(z|D) = "
            f"{_fmt(row['mean_log_variance'], 6)}"
        )
        print(
            f"  mean posterior variance = "
            f"{_fmt(row['mean_posterior_variance'], 6)}"
        )
        print(
            f"  mean posterior SD = "
            f"{_fmt(row['mean_posterior_sd'], 6)}"
        )


    general = subjects[
        subjects["trial_data_name"] == "general_400"
    ].copy()
    if (
        not general.empty
        and general["n_trials"].nunique() > 1
    ):
        rho_logvar = general[
            ["n_trials", "subject_mean_log_variance"]
        ].corr(method="spearman").iloc[0, 1]
        rho_var = general[
            ["n_trials", "subject_mean_posterior_variance"]
        ].corr(method="spearman").iloc[0, 1]

        print("")
        print("Within general_400:")
        print(
            "  Spearman rho(valid trials, subject mean log variance) = "
            f"{_fmt(rho_logvar)}"
        )
        print(
            "  Spearman rho(valid trials, subject posterior variance) = "
            f"{_fmt(rho_var)}"
        )


def make_figure_s6_posterior_concentration():
    subjects, dimensions, summary = build_posterior_diagnostic_frames()
    if summary.empty:
        warnings.warn(
            "Appendix Figure S6 could not be generated: no NP posterior "
            "diagnostics were produced."
        )
        return None

    summary_indexed = summary.set_index("trial_data_name")
    available_order = [
        trial for trial in POSTERIOR_PLOT_ORDER
        if trial in summary_indexed.index
    ]
    x = np.arange(len(available_order), dtype=float)
    labels = [
        POSTERIOR_DESIGN_LABELS[t] for t in available_order
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.75),
        gridspec_kw={"width_ratios": [0.92, 1.03, 1.25]},
    )


    mu_var = [
        float(summary_indexed.loc[t, "mean_mu_variance"])
        for t in available_order
    ]
    axes[0].bar(
        x,
        mu_var,
        width=0.62,
        edgecolor="0.25",
        linewidth=0.5,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel(
        r"Mean across dimensions of $\mathrm{Var}_s(\mu)$"
    )
    axes[0].set_title("Latent spread")
    annotate_panel(axes[0], "A")


    arrays = [
        subjects.loc[
            subjects["trial_data_name"] == trial,
            "subject_mean_log_variance",
        ].dropna().to_numpy()
        for trial in available_order
    ]
    bp = axes[1].boxplot(
        arrays,
        positions=x,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "0.20", "linewidth": 1.0},
        boxprops={"facecolor": "0.88", "edgecolor": "0.35"},
        whiskerprops={"color": "0.45"},
        capprops={"color": "0.45"},
    )
    del bp

    global_means = [
        float(summary_indexed.loc[t, "mean_log_variance"])
        for t in available_order
    ]
    axes[1].scatter(
        x,
        global_means,
        marker="D",
        s=24,
        color=COLORS["np"],
        edgecolors="white",
        linewidths=0.4,
        zorder=4,
        label="Overall mean",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel(r"Subject mean $\log \sigma_z^2$")
    axes[1].set_title("Posterior uncertainty")
    axes[1].legend(
        frameon=False,
        fontsize=6.2,
        loc="best",
    )
    annotate_panel(axes[1], "B")


    general = subjects[
        subjects["trial_data_name"] == "general_400"
    ].copy()
    if (
        not general.empty
        and general["n_trials"].nunique() > 1
    ):
        axes[2].scatter(
            general["n_trials"],
            general["subject_mean_log_variance"],
            s=7,
            alpha=0.16,
            edgecolors="none",
            rasterized=True,
        )


        n_min = int(general["n_trials"].min())
        n_max = int(general["n_trials"].max())
        if n_max > n_min:
            edges = np.linspace(
                n_min,
                n_max + 1,
                num=11,
            )
            general["trial_bin"] = pd.cut(
                general["n_trials"],
                bins=edges,
                include_lowest=True,
                duplicates="drop",
            )
            trend = (
                general.groupby("trial_bin", observed=True)
                .agg(
                    n_trials=("n_trials", "mean"),
                    mean_log_variance=(
                        "subject_mean_log_variance",
                        "mean",
                    ),
                )
                .dropna()
            )
            if not trend.empty:
                axes[2].plot(
                    trend["n_trials"],
                    trend["mean_log_variance"],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.2,
                    color=COLORS["np"],
                )

        rho = general[
            ["n_trials", "subject_mean_log_variance"]
        ].corr(method="spearman").iloc[0, 1]
        axes[2].set_title(
            rf"Full trial-space: $\rho_s={rho:.2f}$"
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "No trial-count variation",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
        )
        axes[2].set_title("Full trial-space")

    axes[2].set_xlabel("Valid trials per subject")
    axes[2].set_ylabel(r"Subject mean $\log \sigma_z^2$")
    annotate_panel(axes[2], "C")

    fig.subplots_adjust(
        wspace=0.42,
        bottom=0.26,
    )

    print_posterior_diagnostic_results(
        subjects,
        summary,
    )
    return fig


CAPTIONS = r"""
Figure numbering below assumes the architecture schematic is Figure 1.

MAIN TEXT

Figure 2. Shared latent geometry reveals simulator relationships and experimental
informativeness. Panels A--B show design-specific latent geometry under the
380- and 27-trial intertemporal-choice designs. The frozen reconstruction-based
NP encoder was trained on OW2--OW5 and AW2 using general_400; within each panel,
PCA axes were fit only to the corresponding representation-training simulations,
and held-out OW1/AW1, the mechanistically distinct H1, and real participants were
projected post hoc. Panel C provides a complementary cross-design view: raw
encoder means from the broad general_400 OW/AW reference simulations and OW1/AW1
from both published designs were jointly projected into one PCA space. Thus
panels A--B use separate design-specific PCA axes, whereas panel C uses a distinct
common PCA reference. Color denotes model family/type and marker/fill encodes
training status or data source.

Figure 3. Simulator-support mismatch and its dependence on experimental design.
For the selected NP representation, panels compare the maximum calibrated
compatibility C_max with either trained family for training simulations,
held-out H1, and real participants. H1 provides a synthetic positive control
for mismatch. The main-text figure intentionally omits comparisons among
representation objectives to preserve space; those checks appear in Appendix
Figure S4.

APPENDIX

Figure S1. Behavioral fidelity of the learned latent geometry. Panels A--B
compare, within each simulator, the Spearman correspondence between pairwise
latent distance and noise-free behavioral-function distance for the
reconstruction-based NP and model-classifier representations under the 380-
and 27-trial designs. Simulators used to train the representations are shown
separately from simulators not used in training. Panel C reports the
model-wise NP-minus-classifier difference.

Figure S2. Exact five-way recovery among the simulators used for representation
learning (OW2--OW5 and AW2). Model-specific latent supports are built from
training simulations, calibrated on validation simulations, and evaluated on
independent test simulations. Overall and per-simulator recovery show where
fine-grained model identity remains identifiable and where simulator variants
become behaviorally degenerate.

Figure S3. Post-hoc coarse-family assignment of held-out OW1 and AW1 under an
encoder trained on OW2--OW5 and AW2. This analysis asks whether held-out
variants generalize to the family-level geometry learned from other simulators;
it also illustrates that mechanistic taxonomy and local behavioral geometry
need not coincide perfectly.

Figure S4. Robustness of simulator-mismatch conclusions. The top row compares
application-versus-training-simulation AUROC across the four architecture-level
representation objectives. The bottom row compares mean Euclidean distance,
local kNN support, and Gaussian negative log-likelihood for the NP
representation. The strong 380-trial mismatch signal generalizes broadly,
whereas the weaker 27-trial signal is more estimator-dependent.

Figure S5. NP generative interpolation / design-compression summary. True
endpoint behavioral-function distance is compared with the distance between
decoder outputs at the corresponding latent endpoints. The 27-trial design can
strongly compress decoded behavioral separation even when the underlying
simulator functions differ substantially.

Figure S6. NP latent spread and posterior uncertainty across the full
trial-space, 380-trial, and 27-trial designs. The same retained OW/AW model
composition (OW1--OW5 and AW1--AW2) and all available test subjects are used in
every design. Panel A reports the mean across latent dimensions of the
between-subject variance of posterior means, Var_s(mu). Panel B shows the
distribution of each subject's mean posterior log variance, with a diamond
marking the overall mean; lower values indicate a more concentrated posterior.
Panel C relates the number of valid trials to posterior log variance within the
full trial-space design, providing a within-design diagnostic of how posterior
uncertainty changes with the amount of behavioral information.
    core, _ = load_core_config()
    rep = core.load_representation(
        POSTERIOR_REPRESENTATION,
        TRAIN_MODELS,
    )
    if rep is None:
        raise FileNotFoundError(
            f"{CFG_NAME} NP checkpoint for general_400 / m4--m8 was not found."
        )

    encoded_groups = []
    for trial in POSTERIOR_DESIGNS:
        experiment = core.load_experiment_data(
            trial,
            data_names=POSTERIOR_DIAGNOSTIC_MODELS,
            seed=SEED,
            full_real=False,
        )

        for model in POSTERIOR_DIAGNOSTIC_MODELS:
            indices = test_indices_for_group(experiment, model)
            if len(indices) == 0:
                continue

            means, logvars, n_trials = infer_np_posterior_parameters(
                core,
                rep,
                experiment,
                indices,
            )
            if len(means) == 0:
                continue

            encoded_groups.append(
                {
                    "trial": trial,
                    "model": model,
                    "indices": indices,
                    "means": means,
                    "logvars": logvars,
                    "n_trials": n_trials,
                }
            )

    if not encoded_groups:
        frame = pd.DataFrame()
        write_source(
            frame,
            "figure_s7_subject_normal_overlays_source.csv",
        )
        return frame

    pooled_means = np.vstack(
        [group["means"] for group in encoded_groups]
    )
    dimension_variances = np.var(pooled_means, axis=0, ddof=0)
    selected_dimension = int(np.argmax(dimension_variances))

    rows = []
    for group in encoded_groups:
        means = group["means"][:, selected_dimension]
        logvars = group["logvars"][:, selected_dimension]
        variances = np.exp(logvars)
        standard_deviations = np.sqrt(variances)

        for i in range(len(means)):
            rows.append(
                {
                    "trial_data_name": group["trial"],
                    "design_label": POSTERIOR_DESIGN_LABELS[
                        group["trial"]
                    ],
                    "model": group["model"],
                    "paper_label": figure_model_label(group["model"]),
                    "combined_subject_index": int(group["indices"][i]),
                    "n_trials": int(group["n_trials"][i]),
                    "selected_z_index": selected_dimension,
                    "selected_z_label": f"z{selected_dimension + 1}",
                    "selected_z_between_subject_variance": float(
                        dimension_variances[selected_dimension]
                    ),
                    "subject_z_mu": float(means[i]),
                    "subject_z_log_variance": float(logvars[i]),
                    "subject_z_posterior_variance": float(variances[i]),
                    "subject_z_posterior_sd": float(
                        standard_deviations[i]
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    write_source(
        frame,
        "figure_s7_subject_normal_overlays_source.csv",
    )
    return frame


def _stable_subject_subset(frame: pd.DataFrame, n_max=220, seed=SEED):
    if len(frame) <= n_max:
        return frame.copy()
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(frame), size=n_max, replace=False))
    return frame.iloc[keep].copy()


def _normal_pdf(x, mu, var):
    var = np.maximum(var, 1e-8)
    return (
        np.exp(-0.5 * (x - mu) ** 2 / var)
        / np.sqrt(2.0 * np.pi * var)
    )


def print_subject_normal_overlay_results(frame: pd.DataFrame):
    print_figure_header("Appendix Figure S7 numeric summary")
    selected_z = str(frame["selected_z_label"].iloc[0])
    selected_variance = float(
        frame["selected_z_between_subject_variance"].iloc[0]
    )
    print(
        f"All panels use {selected_z}, the pooled highest-variance posterior-"
        f"mean dimension (between-subject variance={selected_variance:.6f})."
    )
    for trial in POSTERIOR_PLOT_ORDER:
        sub = frame[frame["trial_data_name"] == trial]
        if sub.empty:
            continue
        print("")
        print(POSTERIOR_DESIGN_LABELS[trial])
        print(f"  n subjects = {len(sub)}")
        print(
            "  mean posterior mu     = "
            f"{_fmt(sub['subject_z_mu'].mean(), 6)}"
        )
        print(
            "  SD of posterior mu    = "
            f"{_fmt(sub['subject_z_mu'].std(ddof=1), 6)}"
        )
        print(
            "  mean posterior var    = "
            f"{_fmt(sub['subject_z_posterior_variance'].mean(), 6)}"
        )
        print(
            "  median posterior var  = "
            f"{_fmt(sub['subject_z_posterior_variance'].median(), 6)}"
        )
        print(
            "  mean posterior SD     = "
            f"{_fmt(sub['subject_z_posterior_sd'].mean(), 6)}"
        )


def make_figure_s7_subject_normal_overlays():
    frame = build_subject_normal_overlay_frame()
    if frame.empty:
        warnings.warn(
            "Appendix Figure S7 could not be generated: no subject-level "
            "posterior summaries were available."
        )
        return None

    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"None of the candidate columns were found: {candidates}")

    mu_col = _pick_col(
        frame,
        [
            "subject_z_mu",
            "subject_mu_on_anchor_dim",
            "posterior_mu_on_anchor_dim",
            "mu_on_anchor_dim",
            "subject_mu_on_z3",
            "posterior_mu_on_z3",
            "subject_mean_mu",
        ],
    )
    var_col = _pick_col(
        frame,
        [
            "subject_z_posterior_variance",
            "subject_posterior_variance_on_anchor_dim",
            "posterior_variance_on_anchor_dim",
            "subject_variance_on_anchor_dim",
            "subject_posterior_variance_on_z3",
            "posterior_variance_on_z3",
            "subject_mean_posterior_variance",
        ],
    )
    sd_col = None
    for c in [
        "subject_z_posterior_sd",
        "subject_posterior_sd_on_anchor_dim",
        "posterior_sd_on_anchor_dim",
        "subject_sd_on_anchor_dim",
        "subject_posterior_sd_on_z3",
        "posterior_sd_on_z3",
        "subject_mean_posterior_sd",
    ]:
        if c in frame.columns:
            sd_col = c
            break

    anchor_label = "selected latent dimension"
    for c in ["anchor_dimension_label", "anchor_dim_label", "selected_dimension_label"]:
        if c in frame.columns and frame[c].notna().any():
            anchor_label = str(frame[c].dropna().iloc[0])
            break
    if (
        anchor_label == "selected latent dimension"
        and "selected_z_label" in frame.columns
        and frame["selected_z_label"].notna().any()
    ):
        anchor_label = str(frame["selected_z_label"].dropna().iloc[0])


    sd_values = (
        frame[sd_col].to_numpy(dtype=float)
        if sd_col is not None
        else np.sqrt(np.maximum(frame[var_col].to_numpy(dtype=float), 1e-8))
    )
    mu_values = frame[mu_col].to_numpy(dtype=float)
    q_low = np.quantile(mu_values - 3.0 * sd_values, 0.01)
    q_high = np.quantile(mu_values + 3.0 * sd_values, 0.99)
    pad = 0.05 * (q_high - q_low if q_high > q_low else 1.0)
    x_grid = np.linspace(q_low - pad, q_high + pad, 900)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 5.35),
        sharex=True,
    )

    panel_colors = {
        "general_400": "0.45",
        "pub_Konstantinidis2020": COLORS["design_380"],
        "pub_Thrailkill2023": COLORS["design_27"],
    }

    all_ymax = []

    for letter, trial, ax in zip("ABC", POSTERIOR_PLOT_ORDER, axes, strict=True):
        sub = frame[frame["trial_data_name"] == trial].copy()
        color = panel_colors.get(trial, "0.35")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if sub.empty:
            ax.set_title(POSTERIOR_DESIGN_LABELS[trial], fontsize=12, pad=8)
            annotate_panel(ax, letter)
            continue

        draw_sub = _stable_subject_subset(
            sub,
            n_max=220 if trial == "general_400" else 180,
            seed=SEED + sum(ord(ch) for ch in trial),
        )

        mus = sub[mu_col].to_numpy(dtype=float)
        vars_ = np.maximum(sub[var_col].to_numpy(dtype=float), 1e-8)
        sds = np.sqrt(vars_)


        indiv_peaks = []
        for _, row in draw_sub.iterrows():
            mu = float(row[mu_col])
            var = max(float(row[var_col]), 1e-8)
            y = _normal_pdf(x_grid, mu, var)
            indiv_peaks.append(float(y.max()))
            ax.plot(
                x_grid,
                y,
                color=color,
                alpha=0.055 if trial == "general_400" else 0.075,
                linewidth=0.6,
                zorder=1,
            )


        mixture = np.zeros_like(x_grid, dtype=float)
        for mu, var in zip(mus, vars_, strict=True):
            mixture += _normal_pdf(x_grid, mu, var)
        mixture /= max(len(sub), 1)

        ax.fill_between(
            x_grid,
            0,
            mixture,
            color=color,
            alpha=0.10,
            zorder=2,
        )
        ax.plot(
            x_grid,
            mixture,
            color=color,
            linewidth=2.2,
            zorder=3,
            label="Average density",
        )


        rug_y = -0.015 * max(mixture.max(), 1e-6)
        rug_sample = _stable_subject_subset(
            sub.sort_values(mu_col),
            n_max=240 if trial == "general_400" else 180,
            seed=SEED + 17 + sum(ord(ch) for ch in trial),
        )
        ax.plot(
            rug_sample[mu_col],
            np.full(len(rug_sample), rug_y),
            linestyle="none",
            marker="o",
            markersize=2.0,
            alpha=0.40,
            color=color,
            zorder=4,
        )


        ax.axvline(0.0, color="0.60", linestyle="--", linewidth=0.9, zorder=0)


        ax.set_title(POSTERIOR_DESIGN_LABELS[trial], fontsize=12, pad=8)
        mean_sd = float(np.mean(sds))
        sd_of_means = float(np.std(mus, ddof=1))
        ax.text(
            0.03,
            0.92,
            rf"$\overline{{\sigma}}={mean_sd:.2f}$" "\n"
            rf"$SD(\bar{{\mu}})={sd_of_means:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
            color="0.25",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.4,
                alpha=0.80,
            ),
        )

        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        annotate_panel(ax, letter)
        all_ymax.append(
            max(
                float(mixture.max()) * 1.28,
                float(np.quantile(indiv_peaks, 0.97)) * 1.08 if indiv_peaks else 0.0,
            )
        )


    ymax = max(all_ymax) if all_ymax else 1.0
    for ax in axes:
        ax.set_ylim(bottom=-0.03 * ymax, top=ymax)
        ax.set_yticks([0, round(ymax / 2, 1), round(ymax, 1)])

    axes[1].set_ylabel("Density", fontsize=12)
    axes[-1].set_xlabel(
        rf"Posterior mean $\mu$ on {anchor_label}",
        fontsize=12,
    )

    fig.suptitle(
        f"Subject-level posterior normals on {anchor_label} "
        "(highest-variance dimension)",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(
        hspace=0.42,
        left=0.07,
        right=0.985,
        bottom=0.12,
        top=0.86,
    )

    print_subject_normal_overlay_results(frame)
    return fig


if __name__ == "__main__":
    try:
        fig = make_figure_s7_subject_normal_overlays()
        if fig is not None:
            save_figure(
                fig,
                f"figure_s7_subject_normal_overlays_{CFG_NAME}",
                "APPENDIX",
                (
                    "Each test subject shown as a 1-D normal on the shared "
                    "highest-variance latent dimension."
                ),
            )
    except Exception as exc:
        warnings.warn(
            "Appendix Figure S7 could not be generated. "
            f"Reason: {type(exc).__name__}: {exc}"
        )
