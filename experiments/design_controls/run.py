from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "design_controls"
SRC = ROOT / "experiments" / "latent_support"
LATENT = ROOT / "experiments" / "representation_learning"
VIZ = ROOT / "experiments" / "visualization"
for p in (ROOT, ROOT / "src", LATENT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common
SOURCE_DESIGN = "pub_Konstantinidis2020"
MODELS = ("m1", "m3", "m4", "m5", "m6", "m7", "m8", "m9")
TRAIN_MODELS = ("m4", "m5", "m6", "m7", "m8")
CANDIDATES = {"OW": ("m4", "m5", "m6", "m7"), "AW": ("m8",)}
DATA_NAMES = ("r1",) + MODELS
NP_KIND = "nn1"
SEED = 31
K = 10
CAP = 1000


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_components():
    core = module(SRC / "step_02_latent_support_analysis.py", "trial_control_support_core")
    cfg = module(SRC / "step_03_final_latent_support_cfgs" / "cfg3.py", "trial_control_cfg3")
    core.REPRESENTATION_TRAIN_TRIAL_DATA_NAME = "general_400"
    core.REPRESENTATION_HPARAMS = dict(cfg.REPRESENTATION_HPARAMS)
    core.MAX_POINTS_PER_DATA = CAP
    core.KNN_K = K
    core.SCORE_KINDS = ("knn",)
    core.SHOW_FIGURES = False
    rep = core.load_representation(NP_KIND, TRAIN_MODELS)
    if rep is None:
        raise FileNotFoundError("Original cfg3 general_400 m4--m8 nn1 checkpoint missing")
    viz = module(VIZ / "step_01_make_paper_figures_cfg3_combined_latent_figure_v5_fixed copy.py", "trial_control_viz")
    viz.SHOW_LATENT_MODEL_LABELS = 0
    viz.TRIAL_LABELS["controlled27"] = "27 trials sampled from 380-trial design"
    return core, rep, viz


def stimulus_subset(n_trials, count, seed):
    if count > n_trials:
        raise ValueError(f"Requested {count} from only {n_trials} trials")
    return np.sort(np.random.default_rng(seed).choice(n_trials, size=count, replace=False))


def load_matched_experiment(indices):
    previous = common.load_choice_data
    subset = None if indices is None else np.asarray(indices, dtype=np.int64)

    def restricted_load(path):
        data = previous(path)
        if subset is None:
            return data
        trial_count = data.trials.shape[-2]
        if subset.max() >= trial_count:
            raise ValueError(f"Out-of-range stimulus for {path}: T={trial_count}")
        idx = torch.as_tensor(subset, dtype=torch.long)
        trials = data.trials.index_select(-2, idx)
        choices = data.choices.index_select(1, idx)
        mask = data.trial_mask.index_select(1, idx)
        return replace(data, trials=trials, choices=choices, trial_mask=mask)

    try:
        common.load_choice_data = restricted_load
        return common.load_experiment_data(
            SOURCE_DESIGN, data_names=DATA_NAMES, seed=SEED, full_real=True
        )
    finally:
        common.load_choice_data = previous


def group_indices(core, experiment, name, split):
    return np.asarray(core.indices_for_data_split(experiment, name, split), dtype=np.int64)


def selected(indices, limit=CAP):
    if limit <= 0 or len(indices) <= limit:
        return indices
    return np.sort(np.random.default_rng(SEED).choice(indices, limit, replace=False))


def knn_dist(reference, query, k=K):
    nn = NearestNeighbors(n_neighbors=min(k, len(reference)), metric="euclidean")
    nn.fit(reference)
    return nn.kneighbors(query, return_distance=True)[0].mean(axis=1)


def calibrated(calibration, query):
    asc = np.sort(np.asarray(calibration, dtype=float))
    return (1 + len(asc) - np.searchsorted(asc, query, side="left")) / (len(asc) + 1)


def auc_ci(positives, negatives, seed, bootstrap=500):
    positives = np.asarray(positives)
    negatives = np.asarray(negatives)
    truth = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    estimate = roc_auc_score(truth, np.r_[positives, negatives])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        pos = rng.choice(positives, len(positives), replace=True)
        neg = rng.choice(negatives, len(negatives), replace=True)
        samples.append(roc_auc_score(truth, np.r_[pos, neg]))
    low, high = np.quantile(samples, [.025, .975])
    return float(estimate), float(low), float(high)


def encode_and_analyze(core, rep, viz, experiment, label, out, seed, bootstrap):
    out.mkdir(parents=True, exist_ok=True)
    z_raw, _ = core.encode_representation(rep, experiment)
    scaler = core.fit_latent_scale(z_raw, experiment, TRAIN_MODELS)
    z = scaler.transform(z_raw)
    indices = {
        model: {split: selected(group_indices(core, experiment, model, split))
                for split in ("train", "validation", "test")}
        for model in MODELS
    }
    real_idx = np.asarray(experiment.subject_indices_by_data["r1"], dtype=np.int64)


    fit_idx = np.concatenate([indices[m]["train"] for m in TRAIN_MODELS])
    pca = PCA(n_components=2, random_state=SEED).fit(z[fit_idx])
    rows = []
    for name in DATA_NAMES:
        idx = real_idx if name == "r1" else indices[name]["test"]
        idx = selected(idx, limit=300)
        xy = pca.transform(z[idx])
        for ii, point in zip(idx, xy):
            rows.append(dict(trial_data_name="controlled27", data_name=name,
                             family=viz.family_for_model(name),
                             status=("real" if name == "r1" else
                                     "training" if name in TRAIN_MODELS else "heldout"),
                             combined_subject_index=int(ii),
                             pc1=float(point[0]), pc2=float(point[1]),
                             pc1_explained_variance=float(pca.explained_variance_ratio_[0]),
                             pc2_explained_variance=float(pca.explained_variance_ratio_[1])))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "figure_2B_latent_points.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    viz.draw_latent_panel(ax, frame, "controlled27")
    fig.tight_layout()
    fig.savefig(out / "figure_2B_controlled27.png", dpi=450)
    fig.savefig(out / "figure_2B_controlled27.pdf")
    plt.close(fig)


    references = {m: z[indices[m]["train"]] for m in TRAIN_MODELS}
    val_family = {}
    for family, models in CANDIDATES.items():
        parts = []
        for model in models:
            q = z[indices[model]["validation"]]
            parts.append(np.min(np.vstack([knn_dist(references[m], q)
                                             for m in models]), axis=0))
        val_family[family] = np.concatenate(parts)

    def evaluate(query_z):
        scores = {}
        for family, members in CANDIDATES.items():
            distance = np.min(np.vstack(
                [knn_dist(references[m], query_z) for m in members]), axis=0)
            scores[family] = calibrated(val_family[family], distance)
        return scores

    records = []
    for name in DATA_NAMES:
        idx = real_idx if name == "r1" else indices[name]["test"]
        s = evaluate(z[idx])
        for ii, ow, aw in zip(idx, s["OW"], s["AW"]):
            records.append(dict(design=label, subject_group=name,
                                subject_index=int(ii), c_ow=float(ow),
                                c_aw=float(aw), c_max=float(max(ow,aw)),
                                best_family="OW" if ow >= aw else "AW"))
    subj = pd.DataFrame(records)
    subj.to_csv(out / "subject_family_support.csv", index=False)
    candidate = subj[subj.subject_group.isin(sum((list(v) for v in CANDIDATES.values()), []))]
    metrics = []
    for query in ("r1", "m9"):
        p = subj[subj.subject_group == query]
        a, low, high = auc_ci(1 - p.c_max, 1 - candidate.c_max,
                              seed=seed, bootstrap=bootstrap)
        metrics.append(dict(design=label, query=query, auroc=a,
                            ci_low=low, ci_high=high,
                            median_c_max=float(p.c_max.median()),
                            fraction_cmax_below_005=float((p.c_max < .05).mean()),
                            candidate_median_c_max=float(candidate.c_max.median())))
    for m, fam in (("m1", "OW"), ("m3", "AW")):
        part = subj[subj.subject_group == m]
        metrics.append(dict(design=label, query=m, family_recovery_accuracy=
                            float((part.best_family == fam).mean())))


    sim_cal = {}
    sim_ref = {}
    sim_test = {}
    for m in MODELS:
        sim_ref[m] = z[indices[m]["train"]]
        sim_test[m] = z[indices[m]["test"]]
        sim_cal[m] = knn_dist(sim_ref[m], z[indices[m]["validation"]])
    pairs = []
    for i, m in enumerate(MODELS):
        for n in MODELS[i + 1:]:
            c_mn = float(calibrated(sim_cal[n], knn_dist(sim_ref[n], sim_test[m])).mean())
            c_nm = float(calibrated(sim_cal[m], knn_dist(sim_ref[m], sim_test[n])).mean())
            pairs.append(dict(design=label, model_a=m, model_b=n,
                              compatibility_a_to_b=c_mn,
                              compatibility_b_to_a=c_nm,
                              symmetric_compatibility=(c_mn+c_nm)/2,
                              pair_type=("H1" if "m9" in (m,n) else
                                         "within_OW" if m in ("m1","m4","m5","m6","m7") and n in ("m1","m4","m5","m6","m7") else
                                         "within_AW" if m in ("m3","m8") and n in ("m3","m8") else "between_OW_AW")))
    pairframe = pd.DataFrame(pairs)
    pairframe.to_csv(out / "symmetric_pairwise.csv", index=False)
    for typ in ("within_OW","within_AW","between_OW_AW","H1"):
        sub = pairframe[pairframe.pair_type == typ]
        metrics.append(dict(design=label, query="pairwise_" + typ,
                            mean_symmetric_compatibility=float(sub.symmetric_compatibility.mean())))


    means, logvars = [], []
    data = replace(experiment.choice_data, trials=core.apply_checkpoint_standardizer(
        experiment.choice_data.trials, rep.checkpoint.get("standardizer")))
    with torch.no_grad():
        all_idx = np.arange(data.n_subjects)
        for start in range(0, len(all_idx), core.INFERENCE_BATCH_SIZE):
            ix = torch.as_tensor(all_idx[start:start+core.INFERENCE_BATCH_SIZE], dtype=torch.long)
            trials = core.batch_trials(data, ix).to(core.DEVICE)
            mu, logvar = rep.model.encode_context(trials,
                data.choices[ix].to(core.DEVICE), data.trial_mask[ix].to(core.DEVICE))
            means.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    mu = np.concatenate(means)
    var = np.exp(np.concatenate(logvars))
    posterior = []
    for name in DATA_NAMES:
        ix = real_idx if name == "r1" else indices[name]["test"]
        posterior.append(dict(design=label, query=name,
                              n=len(ix), mean_posterior_variance=float(var[ix].mean()),
                              mean_squared_posterior_mean=float((mu[ix]**2).mean()),
                              latent_mean_dispersion=float(mu[ix].var(axis=0).mean())))
    pd.DataFrame(posterior).to_csv(out / "posterior_summary.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "summary_metrics.csv", index=False)
    return pd.DataFrame(metrics), pd.DataFrame(posterior)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10,
                        help="Independent 27-of-380 stimulus subsets")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--smoke", action="store_true",
                        help="Run just the first subsample with 30 bootstrap repetitions")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.smoke:
        args.repeats, args.bootstrap = 1, 30
    OUT.mkdir(parents=True, exist_ok=True)
    core, rep, viz = load_components()

    original = common.load_choice_data(common.choice_data_path(SOURCE_DESIGN, "m4"))
    T = int(original.trials.shape[-2])
    if T != 380:
        raise RuntimeError(f"Expected 380 trials, got {T}")
    summaries, posteriors, manifests = [], [], []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        subset = stimulus_subset(T, 27, seed)
        print(f"[{repeat+1}/{args.repeats}] seed={seed}, 27 of {T} trials", flush=True)
        folder = OUT / f"subset_seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trial_index_zero_based":subset,
                      "trial_index_one_based":subset+1}).to_csv(
                          folder / "selected_trial_indices.csv",index=False)
        experiment = load_matched_experiment(subset)
        metrics, posterior = encode_and_analyze(core, rep, viz, experiment,
                       f"sampled27_s{seed}", folder, seed, args.bootstrap)
        summaries.append(metrics.assign(subset_seed=seed))
        posteriors.append(posterior.assign(subset_seed=seed))
        manifests.append(dict(seed=seed, indices=subset.tolist()))

    print("Running unchanged 380-trial paired reference", flush=True)
    full = load_matched_experiment(None)
    metrics, posterior = encode_and_analyze(core, rep, viz, full,
                                "original380", OUT / "original380",
                                args.seed, args.bootstrap)
    summaries.append(metrics.assign(subset_seed=-1))
    posteriors.append(posterior.assign(subset_seed=-1))
    pd.concat(summaries, ignore_index=True).to_csv(OUT / "all_summary_metrics.csv", index=False)
    pd.concat(posteriors, ignore_index=True).to_csv(OUT / "all_posterior_summary.csv", index=False)
    with open(OUT / "manifest.json","w",encoding="utf8") as f:
        json.dump(dict(source_design=SOURCE_DESIGN, num_trials=T,
                       subsample_size=27, source_checkpoint=str(rep.checkpoint_path),
                       training_reused=True, subject_split_seed=SEED,
                       selection_seeds=manifests),f,indent=2)
    print("DONE; all results under", OUT)


if __name__ == "__main__":
    main()
