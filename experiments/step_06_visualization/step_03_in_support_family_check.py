from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]

CFG_NAME = "cfg3"
REPRESENTATION = "nn1"
SCORE = "knn"

TRIAL_DATA_NAMES = (
    "pub_Konstantinidis2020",
    "pub_Thrailkill2023",
)

TRIAL_LABELS = {
    "pub_Konstantinidis2020": "380-trial design",
    "pub_Thrailkill2023": "27-trial design",
}

FAMILY_LABELS = {
    "option_wise": "OW",
    "attribute_wise": "AW",
}

SUPPORT_ROOT_CANDIDATES = (
    PROJECT_ROOT
    / "output"
    / "latent_support_final_analysis"
    / "parallel_m4m8"
    / CFG_NAME,


    PROJECT_ROOT
    / "output"
    / "latent_support_final_analysis"
    / CFG_NAME,
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "step_06_visualization"
    / CFG_NAME
    / "in_support_family_check"
)


def find_support_root() -> Path:
    for root in SUPPORT_ROOT_CANDIDATES:
        probe = (
            root
            / "primary"
            / TRIAL_DATA_NAMES[0]
            / "parallel_m4m8_encoder"
            / "training_family_support"
            / "training_simulation_subjects.csv"
        )
        if probe.exists():
            return root

    tried = "\n".join(f"  - {x}" for x in SUPPORT_ROOT_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate the existing training-family support results.\n"
        "Tried:\n"
        f"{tried}\n\n"
        "Run the latest parallel m4--m8 family-support analysis first."
    )


def subject_file(root: Path, trial_data_name: str) -> Path:
    return (
        root
        / "primary"
        / trial_data_name
        / "parallel_m4m8_encoder"
        / "training_family_support"
        / "training_simulation_subjects.csv"
    )


def as_bool_correct_family(frame: pd.DataFrame) -> pd.Series:


    if {"best_family", "query_true_family"}.issubset(frame.columns):
        valid = (
            frame["best_family"].notna()
            & frame["query_true_family"].notna()
            & (frame["query_true_family"].astype(str) != "")
        )
        out = pd.Series(np.nan, index=frame.index, dtype=float)
        out.loc[valid] = (
            frame.loc[valid, "best_family"].astype(str).to_numpy()
            == frame.loc[valid, "query_true_family"].astype(str).to_numpy()
        ).astype(float)
        return out

    if "correct_family" in frame.columns:
        raw = frame["correct_family"]
        if pd.api.types.is_bool_dtype(raw):
            return raw.astype(float)

        text = raw.astype(str).str.strip().str.lower()
        mapping = {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
        }
        return text.map(mapping)

    raise KeyError(
        "The subject-level file contains neither "
        "{best_family, query_true_family} nor correct_family."
    )


def summarize_group(
    frame: pd.DataFrame,
    trial_data_name: str,
    family: str,
) -> dict:
    values = pd.to_numeric(frame["c_max"], errors="coerce")
    correct = as_bool_correct_family(frame)

    finite_c = np.isfinite(values.to_numpy(dtype=float))
    valid_correct = correct.notna()

    return {
        "trial_data_name": trial_data_name,
        "design_label": TRIAL_LABELS[trial_data_name],
        "family": family,
        "family_label": (
            "All candidate OW/AW"
            if family == "__all__"
            else FAMILY_LABELS.get(family, family)
        ),
        "representation": REPRESENTATION,
        "score": SCORE,
        "n_subjects": int(len(frame)),
        "n_finite_cmax": int(finite_c.sum()),
        "mean_c_max": float(values[finite_c].mean()),
        "median_c_max": float(values[finite_c].median()),
        "sd_c_max": float(values[finite_c].std(ddof=1)),
        "prop_cmax_below_005": float(
            (values[finite_c] < 0.05).mean()
        ),
        "n_family_assignable": int(valid_correct.sum()),
        "correct_family_rate": float(correct[valid_correct].mean()),
    }


def load_and_summarize(
    support_root: Path,
    trial_data_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = subject_file(support_root, trial_data_name)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "representation",
        "score",
        "c_max",
        "query_true_family",
        "best_family",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            f"{path} is missing required columns: {sorted(missing)}"
        )


    frame = frame[
        (frame["representation"].astype(str) == REPRESENTATION)
        & (frame["score"].astype(str) == SCORE)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            f"No rows remain for representation={REPRESENTATION!r}, "
            f"score={SCORE!r} in {path}"
        )


    frame = frame[
        frame["query_true_family"].isin(FAMILY_LABELS)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            f"No candidate-family simulations were found in {path}"
        )

    rows = [
        summarize_group(
            frame,
            trial_data_name,
            family="__all__",
        )
    ]

    for family in ("option_wise", "attribute_wise"):
        sub = frame[frame["query_true_family"] == family].copy()
        if sub.empty:
            continue
        rows.append(
            summarize_group(
                sub,
                trial_data_name,
                family=family,
            )
        )

    return frame, pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    print("")
    print("=" * 78)
    print("IN-SUPPORT FAMILY CHECK")
    print("=" * 78)
    print(
        f"Representation = {REPRESENTATION}; score = {SCORE}; "
        "subjects = independent candidate-family test simulations."
    )

    for trial in TRIAL_DATA_NAMES:
        part = summary[summary["trial_data_name"] == trial]
        if part.empty:
            continue

        print("")
        print(TRIAL_LABELS[trial])
        print("-" * len(TRIAL_LABELS[trial]))

        for family in ("__all__", "option_wise", "attribute_wise"):
            row = part[part["family"] == family]
            if row.empty:
                continue
            row = row.iloc[0]

            label = row["family_label"]
            print(
                f"{label:<21} "
                f"n={int(row['n_subjects']):5d} | "
                f"median Cmax={row['median_c_max']:.3f} | "
                f"mean Cmax={row['mean_c_max']:.3f} | "
                f"correct family={100 * row['correct_family_rate']:.1f}% | "
                f"P(Cmax<.05)={100 * row['prop_cmax_below_005']:.1f}%"
            )

        overall = part[part["family"] == "__all__"].iloc[0]
        print("")
        print("Manuscript-ready sentence:")
        print(
            "  Independent test simulations from the candidate OW/AW families "
            f"showed median C_max={overall['median_c_max']:.3f}, and "
            f"{100 * overall['correct_family_rate']:.1f}% were assigned to "
            f"their generating family under the "
            f"{TRIAL_LABELS[trial]}."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize C_max and correct-family assignment for independent "
            "candidate-family test simulations."
        )
    )
    parser.add_argument(
        "--design",
        choices=("both", "380", "27"),
        default="both",
        help="Which published data-collection design(s) to summarize.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    support_root = find_support_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.design == "380":
        trials = ("pub_Konstantinidis2020",)
    elif args.design == "27":
        trials = ("pub_Thrailkill2023",)
    else:
        trials = TRIAL_DATA_NAMES

    subject_parts = []
    summary_parts = []

    for trial in trials:
        subjects, summary = load_and_summarize(
            support_root,
            trial,
        )
        subject_parts.append(subjects.assign(trial_data_name=trial))
        summary_parts.append(summary)

    subjects_all = pd.concat(subject_parts, ignore_index=True)
    summary_all = pd.concat(summary_parts, ignore_index=True)


    summary_all.to_csv(
        OUTPUT_DIR / "in_support_family_check_by_design_and_family.csv",
        index=False,
    )


    overall = summary_all[
        summary_all["family"] == "__all__"
    ].copy()
    overall.to_csv(
        OUTPUT_DIR / "in_support_family_check_overall.csv",
        index=False,
    )


    subjects_all.to_csv(
        OUTPUT_DIR / "in_support_family_check_subjects.csv",
        index=False,
    )

    print_summary(summary_all)

    print("")
    print("=" * 78)
    print("Saved")
    print("=" * 78)
    print(OUTPUT_DIR / "in_support_family_check_by_design_and_family.csv")
    print(OUTPUT_DIR / "in_support_family_check_overall.csv")
    print(OUTPUT_DIR / "in_support_family_check_subjects.csv")
    print("")
    print(f"Input support root: {support_root}")


if __name__ == "__main__":
    main()
