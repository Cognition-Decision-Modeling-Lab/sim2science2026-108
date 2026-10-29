from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "latent_support"))
sys.path.insert(0, str(ROOT / "experiments" / "representation_learning"))
sys.path.insert(0, str(ROOT / "src"))
import step_02_latent_support_analysis as core
import step_04_latent_geometry_validation as geometry

TRAIN = ("m4", "m5", "m6", "m7", "m8")
MODELS = ("m4", "m5", "m6", "m7", "m8", "m1", "m3", "m9", "r1")
DATASET = "pub_Konstantinidis2020"
OUTPUT = ROOT / "OUTPUTS" / "compare_metrics" / "behavioral_summary_baseline"
FEATURES = ("intercept", "amount_difference", "delay_difference",
            "relative_amount_difference", "relative_delay_difference")


def design_matrix(x):
    x = np.asarray(x, dtype=float)
    if x.shape[1] != 4:
        raise ValueError(f"Expected four intertemporal-choice inputs, got {x.shape}")
    a0, d0, a1, d1 = x.T
    return np.column_stack((
        a1 - a0, d1 - d0,
        (a1 - a0) / (np.abs(a1) + np.abs(a0) + 1e-8),
        (d1 - d0) / (np.abs(d1) + np.abs(d0) + 1e-8),
    ))


def extract_xy(experiment, idx):
    data = experiment.choice_data
    trials = data.trials if data.trials.ndim == 2 else data.trials[idx]
    mask = data.trial_mask[idx].detach().cpu().numpy().astype(bool)
    x = trials.detach().cpu().numpy()[mask]
    y = data.choices[idx].detach().cpu().numpy().reshape(-1)[mask]
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Only binary choice 0/1 supported")
    return x, y.astype(int)


def build_summary(experiment, train_indices, ridge):
    examples = []
    for idx in train_indices[:min(400, len(train_indices))]:
        x, _ = extract_xy(experiment, int(idx))
        examples.append(design_matrix(x))
    pooled = np.concatenate(examples)
    center = pooled.mean(axis=0)
    scale = np.maximum(pooled.std(axis=0), 1e-8)
    data = experiment.choice_data
    vectors = np.zeros((data.n_subjects, 5), dtype=float)
    for i in range(data.n_subjects):
        x, y = extract_xy(experiment, i)
        v = (design_matrix(x) - center) / scale
        if len(np.unique(y)) < 2:
            p = (y.sum() + .5) / (len(y) + 1.)
            vectors[i, 0] = np.log(p / (1 - p))
        else:
            model = LogisticRegression(C=1. / ridge, solver="lbfgs",
                                       max_iter=400, fit_intercept=True)
            model.fit(v, y)
            vectors[i, 0] = model.intercept_[0]
            vectors[i, 1:] = model.coef_[0]

    mu = vectors[train_indices].mean(axis=0)
    sd = vectors[train_indices].std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.)
    return (vectors - mu) / sd, pd.DataFrame(
        {"feature": FEATURES, "trial_input_center": np.r_[np.nan, center],
         "trial_input_scale": np.r_[np.nan, scale],
         "coefficient_center": mu, "coefficient_scale": sd})


def k_nearest(queries, references, k=10, chunk=256):
    result = []
    for start in range(0, len(queries), chunk):
        d = cdist(queries[start:start+chunk], references)
        kk = min(k, len(references))
        result.extend(np.partition(d, kk-1, axis=1)[:, :kk].mean(axis=1))
    return np.asarray(result)


def compatibility(test_distance, cal_distance):
    cal = np.sort(cal_distance)
    return (1 + len(cal) - np.searchsorted(cal, test_distance, side="left")) / (len(cal) + 1)


def support_auc(z, experiment, cap, k):


    ref, calib, test = {}, {}, {}
    for name in TRAIN:
        ref[name] = geometry.core.indices_for_data_split(experiment, name, "train")[:cap]
        calib[name] = geometry.core.indices_for_data_split(experiment, name, "validation")[:cap]
        test[name] = geometry.core.indices_for_data_split(experiment, name, "test")[:cap]
        if min(map(len, (ref[name], calib[name], test[name]))) < k + 1:
            raise ValueError(f"Not enough independent subjects for {name}")
    query = geometry.core.indices_for_data_split(experiment, "m9", "test")[:cap]
    if len(query) < 2:
        raise ValueError("H1 test split missing")

    def score(indices):
        values = []
        for name in TRAIN:
            distance = k_nearest(z[indices], z[ref[name]], k)
            calibration = k_nearest(z[calib[name]], z[ref[name]], k)
            values.append(compatibility(distance, calibration))
        return 1 - np.max(np.stack(values, axis=1), axis=1)

    negative = np.concatenate([score(test[name]) for name in TRAIN])
    positive = score(query)
    labels = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
    scores = np.r_[negative, positive]
    auc = roc_auc_score(labels, scores)
    result = pd.DataFrame({"label_out_of_family": labels.astype(int),
                           "out_of_support_score": scores})
    return float(auc), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--ridge", type=float, default=1.)
    parser.add_argument("--max-per-model", type=int, default=1000)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()
    if args.ridge <= 0 or args.max_per_model < 20 or args.knn_k <= 0:
        parser.error("Invalid ridge, max-per-model, or knn-k")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    geometry.configure_core_module()

    experiment = core.load_experiment_data(
        args.dataset, data_names=MODELS, seed=31, full_real=True)
    train_ids = np.concatenate([
        core.indices_for_data_split(experiment, name, "train")
        for name in TRAIN
    ])
    summary, scales = build_summary(experiment, train_ids, args.ridge)
    scales.to_csv(OUTPUT / "summary_feature_scaling.csv", index=False)
    np.save(OUTPUT / "summary_vectors.npy", summary)


    rep = core.load_representation("nn1", TRAIN)
    if rep is None:
        raise FileNotFoundError("Original nn1 checkpoint missing")
    raw, _ = core.encode_representation(rep, experiment)
    latent = core.fit_latent_scale(raw, experiment, TRAIN).transform(raw)


    grid = geometry.common_trial_grid(experiment)
    subset = {}
    for label, vectors in (("behavioral_summary", summary), ("nn1", latent)):
        subset[label] = geometry.RepresentationOutput(
            label=label, kind="baseline" if label == "behavioral_summary" else "nn1",
            rep=rep, latent_raw=vectors, latent_scaled=vectors)


    original_scopes = geometry.geometry_scopes_for_protocol
    def paper_scopes(protocol_name, train_models):
        scopes = original_scopes(protocol_name, train_models)
        return [(scope, group, tuple(m for m in models if m in MODELS))
                for scope, group, models in scopes
                if any(m in MODELS for m in models)
                and not (scope == "within_model" and "m2" in models)]
    try:
        geometry.geometry_scopes_for_protocol = paper_scopes
        geom, _, _ = geometry.run_simulated_geometry(
            args.dataset, "generalization_m4_m8", TRAIN,
            experiment, grid, subset, None)
    finally:
        geometry.geometry_scopes_for_protocol = original_scopes
    geom.to_csv(OUTPUT / "behavioral_fidelity.csv", index=False)
    primary = geom[(geom.scope == "within_model") &
                   (geom.behavior_distance == "functional_rmse")]
    primary.to_csv(OUTPUT / "behavioral_fidelity_primary.csv", index=False)

    out = []
    for label, values in (("behavioral_summary", summary), ("nn1", latent)):
        auc, subjects = support_auc(values, experiment, args.max_per_model,
                                    args.knn_k)
        subjects.to_csv(OUTPUT / f"h1_subject_scores_{label}.csv", index=False)
        out.append({"representation": label, "h1_auroc": auc,
                    "dataset": args.dataset, "knn_k": args.knn_k,
                    "max_per_model": args.max_per_model})
    pd.DataFrame(out).to_csv(OUTPUT / "h1_auroc.csv", index=False)
    print("Saved independent baseline results in:", OUTPUT)
    print(pd.DataFrame(out).to_string(index=False))
    print(primary[["models", "representation", "spearman_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
