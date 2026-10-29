from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import run as base

OW = ("m4", "m5", "m6", "m7")
AW = ("m8",)
HELD_OUT = {"m1": "OW", "m3": "AW"}
FAMILIES = {"OW": OW, "AW": AW}


def select(experiment, model, split, cap):
    indices = base.core.indices_for_data_split(experiment, model, split)
    return np.asarray(indices, dtype=np.int64)[:cap]


def calibrated_recovery(z, experiment, cap, k):
    reference, calibration = {}, {}
    for model in base.TRAIN:
        reference[model] = select(experiment, model, "train", cap)
        calibration[model] = select(experiment, model, "validation", cap)
        if len(reference[model]) < k or len(calibration[model]) < 2:
            raise ValueError(f"Insufficient reference/calibration points for {model}")

    calibrated = {
        model: base.k_nearest(z[calibration[model]], z[reference[model]], k)
        for model in base.TRAIN
    }
    records = []
    for query_model, true_family in HELD_OUT.items():
        query_ids = select(experiment, query_model, "test", cap)
        if len(query_ids) < 2:
            raise ValueError(f"Insufficient test subjects for {query_model}")
        model_c = {
            model: base.compatibility(
                base.k_nearest(z[query_ids], z[reference[model]], k),
                calibrated[model],
            )
            for model in base.TRAIN
        }
        family_c = {
            name: np.max(np.column_stack([model_c[m] for m in members]), axis=1)
            for name, members in FAMILIES.items()
        }
        for index, combined_idx in enumerate(query_ids):
            predicted = "OW" if family_c["OW"][index] >= family_c["AW"][index] else "AW"
            records.append({
                "query_model": query_model,
                "true_family": true_family,
                "predicted_family": predicted,
                "combined_subject_index": int(combined_idx),
                "compatibility_OW": float(family_c["OW"][index]),
                "compatibility_AW": float(family_c["AW"][index]),
                "correct": int(predicted == true_family),
            })
    subject = pd.DataFrame(records)
    accuracy = {model: float(subject.loc[subject.query_model == model, "correct"].mean())
                for model in HELD_OUT}
    balanced = balanced_accuracy_score(subject.true_family, subject.predicted_family)
    matrix = confusion_matrix(subject.true_family, subject.predicted_family, labels=["OW", "AW"])
    return subject, {
        "OW1_accuracy": accuracy["m1"],
        "AW1_accuracy": accuracy["m3"],
        "balanced_accuracy": float(balanced),
        "n_OW1": int((subject.query_model == "m1").sum()),
        "n_AW1": int((subject.query_model == "m3").sum()),
        "trueOW_predOW": int(matrix[0, 0]),
        "trueOW_predAW": int(matrix[0, 1]),
        "trueAW_predOW": int(matrix[1, 0]),
        "trueAW_predAW": int(matrix[1, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=base.DATASET)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--max-per-model", type=int, default=1000)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()
    if args.ridge <= 0 or args.max_per_model < 20 or args.knn_k < 1:
        parser.error("Invalid --ridge, --max-per-model or --knn-k.")

    base.geometry.configure_core_module()
    experiment = base.core.load_experiment_data(
        args.dataset, data_names=base.MODELS, seed=31, full_real=True
    )


    outdir = base.OUTPUT
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "summary_vectors.npy"
    meta = outdir / "family_recovery_feature_cache.json"
    manifest = {"dataset": args.dataset, "ridge": args.ridge,
                "n_subjects": experiment.choice_data.n_subjects}
    can_reuse = False
    if cache.exists() and args.ridge == 1.0 and args.dataset == base.DATASET:
        candidate = np.load(cache, mmap_mode="r")
        can_reuse = candidate.shape == (experiment.choice_data.n_subjects, 5)
    if can_reuse:
        summary = np.asarray(candidate)
        print(f"Reusing existing summary vectors: {cache}")
    else:
        train_ids = np.concatenate([
            select(experiment, name, "train", 10**9) for name in base.TRAIN
        ])
        summary, _ = base.build_summary(experiment, train_ids, args.ridge)
        print("Recomputed behavioral summaries in memory (existing files unchanged).")

    rep = base.core.load_representation("nn1", base.TRAIN)
    if rep is None:
        raise FileNotFoundError("Missing original nn1 checkpoint.")
    raw, _ = base.core.encode_representation(rep, experiment)
    np_latent = base.core.fit_latent_scale(raw, experiment, base.TRAIN).transform(raw)

    rows = []
    for name, vectors in (("behavioral_summary", summary), ("nn1", np_latent)):
        subjects, metrics = calibrated_recovery(
            vectors, experiment, args.max_per_model, args.knn_k
        )
        subjects.insert(0, "representation", name)
        subjects.to_csv(outdir / f"family_recovery_subjects_{name}.csv", index=False)
        rows.append({
            "representation": name,
            "dataset": args.dataset,
            "knn_k": args.knn_k,
            "max_per_model": args.max_per_model,
            "family_aggregation": "max_model_compatibility",
            **metrics,
        })
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "family_recovery_summary.csv", index=False)
    print("\nHeld-out family recovery (identical split/metric across representations):")
    print(result[["representation", "OW1_accuracy", "AW1_accuracy", "balanced_accuracy",
                  "n_OW1", "n_AW1"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
