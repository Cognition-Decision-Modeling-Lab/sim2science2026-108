from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
STEP05_DIR = PROJECT_ROOT / "experiments" / "latent_support"

CFG_NAME = "cfg3"
CFG_DIR = STEP05_DIR / "configs"
CFG_PATH = CFG_DIR / f"{CFG_NAME}.py"

REPRESENTATION = "nn1"
REPRESENTATION_TRAIN_TRIAL_DATA_NAME = "general_400"
REPRESENTATION_TRAIN_MODELS = ("m4", "m5", "m6", "m7", "m8")


MODELS = ("m1", "m4", "m5", "m6", "m7", "m3", "m8", "m9")

TRIAL_DATA_NAMES = (
    "pub_Konstantinidis2020",
    "pub_Thrailkill2023",
)
TRIAL_LABELS = {
    "pub_Konstantinidis2020": "380-trial design",
    "pub_Thrailkill2023": "27-trial design",
}
TRIAL_SLUGS = {
    "pub_Konstantinidis2020": "380_trial",
    "pub_Thrailkill2023": "27_trial",
}

SEED = 31
KNN_K = 10
MAX_POINTS_PER_DATA = 1000
BOOTSTRAP_RESAMPLES = 1000

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "visualization"
    / CFG_NAME
    / "pairwise_model_compatibility"
)


sys.path.insert(0, str(PROJECT_ROOT / "src"))

from choice_generators.model_name_transfer import (
    paper_model_family,
    paper_model_label,
)


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


def load_core_and_cfg():
    latent_dir = PROJECT_ROOT / "experiments" / "representation_learning"
    for path in (PROJECT_ROOT, PROJECT_ROOT / "src", latent_dir, STEP05_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    core_path = STEP05_DIR / "support_core.py"
    core = load_module(core_path, "step06_pairwise_support_core")
    cfg = load_module(CFG_PATH, f"step06_pairwise_{CFG_NAME}")

    if getattr(cfg, "CONFIG_NAME", None) != CFG_NAME:
        raise ValueError(
            f"Expected CONFIG_NAME={CFG_NAME!r}, got "
            f"{getattr(cfg, 'CONFIG_NAME', None)!r}"
        )


    core.REPRESENTATION_TRAIN_TRIAL_DATA_NAME = (
        REPRESENTATION_TRAIN_TRIAL_DATA_NAME
    )
    core.REPRESENTATION_HPARAMS = dict(cfg.REPRESENTATION_HPARAMS)
    core.MAX_POINTS_PER_DATA = int(MAX_POINTS_PER_DATA)
    core.KNN_K = int(KNN_K)
    core.SCORE_KINDS = ("knn",)

    return core, cfg


def knn_mean_distance(
    reference_z: np.ndarray,
    query_z: np.ndarray,
    k: int,
) -> np.ndarray:
    reference_z = np.asarray(reference_z, dtype=np.float64)
    query_z = np.asarray(query_z, dtype=np.float64)

    if len(reference_z) == 0:
        raise ValueError("Reference cloud is empty.")
    if len(query_z) == 0:
        return np.empty(0, dtype=np.float64)

    k_eff = min(max(1, int(k)), len(reference_z))
    nn = NearestNeighbors(
        n_neighbors=k_eff,
        metric="euclidean",
        algorithm="auto",
    )
    nn.fit(reference_z)
    distances, _ = nn.kneighbors(query_z, return_distance=True)
    return distances.mean(axis=1)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[b] = sample.mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_symmetric_ci(
    values_ab: np.ndarray,
    values_ba: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    values_ab = np.asarray(values_ab, dtype=np.float64)
    values_ba = np.asarray(values_ba, dtype=np.float64)
    values_ab = values_ab[np.isfinite(values_ab)]
    values_ba = values_ba[np.isfinite(values_ba)]

    if len(values_ab) == 0 or len(values_ba) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        a = rng.choice(values_ab, size=len(values_ab), replace=True).mean()
        bval = rng.choice(values_ba, size=len(values_ba), replace=True).mean()
        estimates[b] = 0.5 * (a + bval)

    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def pair_category(model_a: str, model_b: str) -> str:
    fam_a = paper_model_family(model_a)
    fam_b = paper_model_family(model_b)

    if fam_a == "OW" and fam_b == "OW":
        return "within_OW"
    if fam_a == "AW" and fam_b == "AW":
        return "within_AW"
    if {fam_a, fam_b} == {"OW", "AW"}:
        return "OW_vs_AW"
    if "H" in {fam_a, fam_b} and "OW" in {fam_a, fam_b}:
        return "H1_vs_OW"
    if "H" in {fam_a, fam_b} and "AW" in {fam_a, fam_b}:
        return "H1_vs_AW"
    return "other"


def design_seed(trial_data_name: str, *parts: str) -> int:
    token = "|".join((trial_data_name, *parts))
    return SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(token))


@dataclass
class DesignResult:
    trial_data_name: str
    subjects: pd.DataFrame
    directional: pd.DataFrame
    symmetric: pd.DataFrame
    category_summary: pd.DataFrame


def analyze_design(core, trial_data_name: str) -> DesignResult:
    print("")
    print("=" * 80)
    print(TRIAL_LABELS[trial_data_name])
    print("=" * 80)


    core.TRIAL_DATA_NAME = trial_data_name

    experiment = core.load_experiment_data(
        trial_data_name,
        data_names=MODELS,
        seed=SEED,
        full_real=False,
    )

    rep = core.load_representation(
        REPRESENTATION,
        REPRESENTATION_TRAIN_MODELS,
    )
    if rep is None:
        raise FileNotFoundError(
            f"Could not find {CFG_NAME} {REPRESENTATION} checkpoint trained on "
            f"{REPRESENTATION_TRAIN_TRIAL_DATA_NAME} / "
            f"{REPRESENTATION_TRAIN_MODELS}."
        )

    latent, _ = core.encode_representation(rep, experiment)


    latent_scale = core.fit_latent_scale(
        latent,
        experiment,
        REPRESENTATION_TRAIN_MODELS,
    )
    z = latent_scale.transform(latent)


    test_indices = {
        model: np.asarray(
            core.indices_for_data_split(experiment, model, "test"),
            dtype=np.int64,
        )
        for model in MODELS
    }

    subject_rows: list[dict] = []


    for candidate_model in MODELS:
        ref_idx = np.asarray(
            core.indices_for_data_split(
                experiment,
                candidate_model,
                "train",
            ),
            dtype=np.int64,
        )
        cal_idx = np.asarray(
            core.indices_for_data_split(
                experiment,
                candidate_model,
                "validation",
            ),
            dtype=np.int64,
        )

        if len(ref_idx) < 2 or len(cal_idx) < 1:
            raise ValueError(
                f"{candidate_model}: need nonempty train/reference and "
                "validation/calibration splits."
            )

        z_ref = z[ref_idx]
        z_cal = z[cal_idx]


        k_eff = min(max(1, int(KNN_K)), len(z_ref))
        nn = NearestNeighbors(
            n_neighbors=k_eff,
            metric="euclidean",
            algorithm="auto",
        )
        nn.fit(z_ref)

        cal_dist, _ = nn.kneighbors(z_cal, return_distance=True)
        cal_raw = cal_dist.mean(axis=1)

        for query_model in MODELS:
            q_idx = test_indices[query_model]
            if len(q_idx) == 0:
                continue

            q_dist, _ = nn.kneighbors(z[q_idx], return_distance=True)
            q_raw = q_dist.mean(axis=1)

            compatibility = core.compatibility_from_calibration(
                cal_raw,
                q_raw,
            )

            src_idx = core.source_indices(experiment, q_idx)

            for local_i in range(len(q_idx)):
                subject_rows.append(
                    {
                        "trial_data_name": trial_data_name,
                        "design_label": TRIAL_LABELS[trial_data_name],
                        "representation": REPRESENTATION,
                        "query_model": query_model,
                        "query_label": paper_model_label(query_model),
                        "query_family": paper_model_family(query_model),
                        "candidate_model": candidate_model,
                        "candidate_label": paper_model_label(candidate_model),
                        "candidate_family": paper_model_family(candidate_model),
                        "combined_subject_index": int(q_idx[local_i]),
                        "source_subject_index": int(src_idx[local_i]),
                        "knn_k": int(k_eff),
                        "raw_knn_distance": float(q_raw[local_i]),
                        "compatibility": float(compatibility[local_i]),
                    }
                )

    subjects = pd.DataFrame(subject_rows)


    directional_rows = []
    for (query_model, candidate_model), group in subjects.groupby(
        ["query_model", "candidate_model"],
        sort=False,
    ):
        values = group["compatibility"].to_numpy(dtype=np.float64)
        raw = group["raw_knn_distance"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            n_boot=BOOTSTRAP_RESAMPLES,
            seed=design_seed(
                trial_data_name,
                query_model,
                candidate_model,
                "directional",
            ),
        )

        directional_rows.append(
            {
                "trial_data_name": trial_data_name,
                "design_label": TRIAL_LABELS[trial_data_name],
                "query_model": query_model,
                "query_label": paper_model_label(query_model),
                "query_family": paper_model_family(query_model),
                "candidate_model": candidate_model,
                "candidate_label": paper_model_label(candidate_model),
                "candidate_family": paper_model_family(candidate_model),
                "n_query_subjects": int(np.isfinite(values).sum()),
                "mean_compatibility": float(np.nanmean(values)),
                "median_compatibility": float(np.nanmedian(values)),
                "sd_compatibility": float(np.nanstd(values, ddof=1)),
                "mean_compatibility_ci_low": ci_low,
                "mean_compatibility_ci_high": ci_high,
                "mean_raw_knn_distance": float(np.nanmean(raw)),
            }
        )

    directional = pd.DataFrame(directional_rows)


    value_lookup = {
        (q, c): g["compatibility"].to_numpy(dtype=np.float64)
        for (q, c), g in subjects.groupby(
            ["query_model", "candidate_model"],
            sort=False,
        )
    }
    mean_lookup = {
        (row.query_model, row.candidate_model): float(row.mean_compatibility)
        for row in directional.itertuples(index=False)
    }


    symmetric_rows = []
    for i, model_a in enumerate(MODELS):
        for j in range(i, len(MODELS)):
            model_b = MODELS[j]

            mean_ab = mean_lookup[(model_a, model_b)]
            mean_ba = mean_lookup[(model_b, model_a)]
            similarity = 0.5 * (mean_ab + mean_ba)

            self_a = mean_lookup[(model_a, model_a)]
            self_b = mean_lookup[(model_b, model_b)]


            deficit = 0.5 * (
                (self_a - mean_ab)
                + (self_b - mean_ba)
            )

            if model_a == model_b:
                ci_low, ci_high = bootstrap_mean_ci(
                    value_lookup[(model_a, model_a)],
                    n_boot=BOOTSTRAP_RESAMPLES,
                    seed=design_seed(
                        trial_data_name,
                        model_a,
                        model_b,
                        "symmetric",
                    ),
                )
            else:
                ci_low, ci_high = bootstrap_symmetric_ci(
                    value_lookup[(model_a, model_b)],
                    value_lookup[(model_b, model_a)],
                    n_boot=BOOTSTRAP_RESAMPLES,
                    seed=design_seed(
                        trial_data_name,
                        model_a,
                        model_b,
                        "symmetric",
                    ),
                )

            symmetric_rows.append(
                {
                    "trial_data_name": trial_data_name,
                    "design_label": TRIAL_LABELS[trial_data_name],
                    "model_a": model_a,
                    "label_a": paper_model_label(model_a),
                    "family_a": paper_model_family(model_a),
                    "model_b": model_b,
                    "label_b": paper_model_label(model_b),
                    "family_b": paper_model_family(model_b),
                    "pair_category": (
                        "self"
                        if model_a == model_b
                        else pair_category(model_a, model_b)
                    ),
                    "mean_compatibility_a_to_b": mean_ab,
                    "mean_compatibility_b_to_a": mean_ba,
                    "symmetric_compatibility": similarity,
                    "symmetric_compatibility_ci_low": ci_low,
                    "symmetric_compatibility_ci_high": ci_high,
                    "self_compatibility_a": self_a,
                    "self_compatibility_b": self_b,
                    "compatibility_deficit": deficit,
                }
            )

    symmetric = pd.DataFrame(symmetric_rows)


    offdiag = symmetric[symmetric["model_a"] != symmetric["model_b"]].copy()
    category_summary = (
        offdiag.groupby("pair_category", as_index=False)
        .agg(
            n_model_pairs=("symmetric_compatibility", "size"),
            mean_symmetric_compatibility=(
                "symmetric_compatibility",
                "mean",
            ),
            median_symmetric_compatibility=(
                "symmetric_compatibility",
                "median",
            ),
            mean_compatibility_deficit=(
                "compatibility_deficit",
                "mean",
            ),
        )
    )
    category_summary.insert(0, "trial_data_name", trial_data_name)
    category_summary.insert(1, "design_label", TRIAL_LABELS[trial_data_name])


    known = offdiag[
        offdiag["pair_category"].isin(["within_OW", "within_AW"])
    ].copy()
    known["family_relation"] = "within_family"
    known_summary = (
        known.groupby("family_relation", as_index=False)
        .agg(
            n_model_pairs=("symmetric_compatibility", "size"),
            mean_symmetric_compatibility=(
                "symmetric_compatibility",
                "mean",
            ),
            median_symmetric_compatibility=(
                "symmetric_compatibility",
                "median",
            ),
            mean_compatibility_deficit=(
                "compatibility_deficit",
                "mean",
            ),
        )
        .rename(columns={"family_relation": "pair_category"})
    )
    known_summary.insert(0, "trial_data_name", trial_data_name)
    known_summary.insert(1, "design_label", TRIAL_LABELS[trial_data_name])

    category_summary = pd.concat(
        [category_summary, known_summary],
        ignore_index=True,
    )

    print_directional_matrix(directional, trial_data_name)
    print_symmetric_summary(symmetric, category_summary, trial_data_name)

    return DesignResult(
        trial_data_name=trial_data_name,
        subjects=subjects,
        directional=directional,
        symmetric=symmetric,
        category_summary=category_summary,
    )


def matrix_from_symmetric(
    symmetric: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    labels = [paper_model_label(m) for m in MODELS]
    matrix = pd.DataFrame(
        np.nan,
        index=labels,
        columns=labels,
        dtype=float,
    )

    for row in symmetric.itertuples(index=False):
        a = paper_model_label(row.model_a)
        b = paper_model_label(row.model_b)
        value = float(getattr(row, value_col))
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value

    return matrix


def print_directional_matrix(
    directional: pd.DataFrame,
    trial_data_name: str,
):
    matrix = directional.pivot(
        index="query_label",
        columns="candidate_label",
        values="mean_compatibility",
    )
    labels = [paper_model_label(m) for m in MODELS]
    matrix = matrix.reindex(index=labels, columns=labels)

    print("")
    print(
        "Directional mean calibrated compatibility C(A -> B); "
        "rows=query A, columns=candidate B"
    )
    print(matrix.round(3).to_string())


def print_symmetric_summary(
    symmetric: pd.DataFrame,
    category_summary: pd.DataFrame,
    trial_data_name: str,
):
    matrix = matrix_from_symmetric(
        symmetric,
        "symmetric_compatibility",
    )

    print("")
    print("Symmetric pairwise compatibility S(A,B)")
    print(matrix.round(3).to_string())

    print("")
    print("Pair-category summary")
    cols = [
        "pair_category",
        "n_model_pairs",
        "mean_symmetric_compatibility",
        "mean_compatibility_deficit",
    ]
    print(category_summary[cols].round(3).to_string(index=False))


    lookup = category_summary.set_index("pair_category")
    print("")
    print("Paper-oriented contrasts:")
    if {"within_family", "OW_vs_AW"}.issubset(lookup.index):
        within = float(
            lookup.loc["within_family", "mean_symmetric_compatibility"]
        )
        between = float(
            lookup.loc["OW_vs_AW", "mean_symmetric_compatibility"]
        )
        print(
            "  Mean compatibility within OW/AW families "
            f"= {within:.3f}; OW-vs-AW = {between:.3f}; "
            f"difference = {within - between:.3f}."
        )

    for category in ("H1_vs_OW", "H1_vs_AW"):
        if category in lookup.index:
            value = float(
                lookup.loc[category, "mean_symmetric_compatibility"]
            )
            print(f"  {category}: mean compatibility = {value:.3f}.")


def save_heatmap(
    matrix: pd.DataFrame,
    trial_data_name: str,
    output_dir: Path,
    show: bool,
):
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Symmetric calibrated compatibility",
    )

    labels = list(matrix.index)
    x = np.arange(len(labels))
    ax.set_xticks(x)
    ax.set_yticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Simulator")
    ax.set_ylabel("Simulator")
    ax.set_title(TRIAL_LABELS[trial_data_name])

    values = matrix.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.tight_layout()
    slug = TRIAL_SLUGS[trial_data_name]
    fig.savefig(
        output_dir / f"pairwise_similarity_heatmap_{slug}.png",
        dpi=300,
    )
    fig.savefig(
        output_dir / f"pairwise_similarity_heatmap_{slug}.pdf",
    )

    if show:
        plt.show()
    plt.close(fig)


def write_design_outputs(result: DesignResult, show: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = TRIAL_SLUGS[result.trial_data_name]

    result.subjects.to_csv(
        OUTPUT_DIR / f"pairwise_subject_compatibility_{slug}.csv",
        index=False,
    )
    result.directional.to_csv(
        OUTPUT_DIR / f"directional_compatibility_{slug}.csv",
        index=False,
    )
    result.symmetric.to_csv(
        OUTPUT_DIR / f"symmetric_pairwise_{slug}.csv",
        index=False,
    )
    result.category_summary.to_csv(
        OUTPUT_DIR / f"pair_category_summary_{slug}.csv",
        index=False,
    )

    similarity_matrix = matrix_from_symmetric(
        result.symmetric,
        "symmetric_compatibility",
    )
    deficit_matrix = matrix_from_symmetric(
        result.symmetric,
        "compatibility_deficit",
    )

    similarity_matrix.to_csv(
        OUTPUT_DIR / f"symmetric_similarity_matrix_{slug}.csv"
    )
    deficit_matrix.to_csv(
        OUTPUT_DIR / f"compatibility_deficit_matrix_{slug}.csv"
    )

    save_heatmap(
        similarity_matrix,
        result.trial_data_name,
        OUTPUT_DIR,
        show=show,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute calibrated pairwise simulator compatibility in the cfg3 "
            "NP latent space."
        )
    )
    parser.add_argument(
        "--design",
        choices=("both", "380", "27"),
        default="both",
        help="Which published data-collection design(s) to analyze.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show heatmaps interactively in addition to saving them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.design == "380":
        trials = ("pub_Konstantinidis2020",)
    elif args.design == "27":
        trials = ("pub_Thrailkill2023",)
    else:
        trials = TRIAL_DATA_NAMES

    core, _ = load_core_and_cfg()

    results = []
    for trial in trials:
        result = analyze_design(core, trial)
        write_design_outputs(result, show=args.show)
        results.append(result)


    pd.concat(
        [r.directional for r in results],
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "directional_compatibility_all_designs.csv",
        index=False,
    )
    pd.concat(
        [r.symmetric for r in results],
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "symmetric_pairwise_all_designs.csv",
        index=False,
    )
    pd.concat(
        [r.category_summary for r in results],
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "pair_category_summary_all_designs.csv",
        index=False,
    )

    print("")
    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Representation: {REPRESENTATION} ({CFG_NAME})")
    print(f"Encoder training models: {REPRESENTATION_TRAIN_MODELS}")
    print(f"kNN k: {KNN_K}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(
        "Primary paper-facing quantity: symmetric_compatibility; "
        "larger values indicate greater model--model similarity."
    )


if __name__ == "__main__":
    main()
