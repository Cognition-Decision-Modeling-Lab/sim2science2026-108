from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "robustness" / "support_sensitivity"
SUPPORT = ROOT / "output" / "latent_support_final_analysis" / "parallel_m4m8" / "cfg3"
CORE_PATH = ROOT / "experiments" / "latent_support" / "support_core.py"
CFG_PATH = ROOT / "experiments" / "latent_support" / "configs" / "cfg3.py"
DESIGNS = ("pub_Konstantinidis2020", "pub_Thrailkill2023")
OW = ("m4", "m5", "m6", "m7")
AW = ("m8",)
TRAIN = OW + AW
K_VALUES = (5, 10, 20, 40)
SEEDS = (31, 32, 33)


def write(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


def require(frame, columns, source):
    absent = set(columns) - set(frame.columns)
    if absent:
        raise ValueError(f"{source}: missing columns {sorted(absent)}")


def recompute_missing_sensitivity():
    core = make_core()
    rep = core.load_representation("nn1", TRAIN)
    if rep is None:
        raise FileNotFoundError("Missing cfg3 general_400 m4--m8 nn1 checkpoint")
    rows = []
    for design in DESIGNS:
        seed_list = SEEDS if design == DESIGNS[0] else (31,)
        for seed in seed_list:
            print(f"Encoding {design}, partition seed={seed}", flush=True)
            core.TRIAL_DATA_NAME = design
            core.SEED = seed
            exp = core.load_experiment_data(
                design, data_names=("r1", "m9") + TRAIN,
                seed=seed, full_real=True)
            latent, _ = core.encode_representation(rep, exp)
            scale = core.fit_latent_scale(latent, exp, TRAIN)
            z = np.asarray(scale.transform(latent), dtype=float)
            for k in (K_VALUES if seed == 31 else (10,)):
                knn = {}
                for model in TRAIN:
                    r_idx = idx(core, exp, model, "train")
                    if len(r_idx) < k:
                        raise ValueError(f"Insufficient reference points: {model}")
                    knn[model] = NearestNeighbors(
                        n_neighbors=k, metric="euclidean").fit(z[r_idx])

                def distance(models, points):
                    return np.min(np.column_stack([
                        knn[m].kneighbors(points, return_distance=True)[0].mean(axis=1)
                        for m in models]), axis=1)

                family_models = {"option_wise": OW, "attribute_wise": AW}
                calibration = {}
                for family, models in family_models.items():
                    indices = np.concatenate([
                        idx(core, exp, m, "validation") for m in models])
                    calibration[family] = distance(models, z[indices])

                def cmax(points):
                    return np.maximum(*[
                        core.compatibility_from_calibration(
                            calibration[f], distance(models, points))
                        for f, models in family_models.items()])

                sim_scores = np.concatenate([
                    cmax(z[idx(core, exp, m, "test")]) for m in TRAIN])
                for query in ("r1", "m9"):
                    q_idx = (core.all_indices_for_data(exp, query)
                             if query == "r1" else idx(core, exp, query, "test"))
                    app_scores = cmax(z[q_idx])
                    auc = roc_auc_score(
                        np.r_[np.zeros(len(sim_scores)), np.ones(len(app_scores))],
                        np.r_[1 - sim_scores, 1 - app_scores])
                    sweeps = []
                    if seed == 31:
                        sweeps.append(("k", k, 31, k))
                    if k == 10:
                        sweeps.append(("split_seed", 10, seed, seed))
                    for sweep, kk, sd, condition in sweeps:
                        rows.append(dict(sweep=sweep, trial_data_name=design,
                            application_query=query, knn_k=kk, split_seed=sd,
                            condition=condition, auroc=float(auc),
                            n_real=len(app_scores), n_simulation=len(sim_scores)))
                print(f"  k={k}: completed", flush=True)
    full = pd.DataFrame(rows)
    write(full, "01_auroc_sensitivity_individual.csv")
    summary = full.groupby(
        ["sweep", "trial_data_name", "application_query"], as_index=False
    ).agg(n_conditions=("auroc", "size"), mean_auroc=("auroc", "mean"),
          min_auroc=("auroc", "min"), max_auroc=("auroc", "max"),
          sd_auroc=("auroc", "std"))
    write(summary, "02_auroc_sensitivity_summary.csv")
    return summary


def sensitivity():
    a = SUPPORT / "aggregate"
    paths = {
        "k": a / "knn_robustness_all.csv",
        "split_seed": a / "split_seed_robustness_all.csv",
    }
    if any(not path.is_file() for path in paths.values()):
        print("Original aggregate CSVs missing; recomputing downstream AUROC from frozen cfg3 NP.", flush=True)
        return recompute_missing_sensitivity()
    parts = []
    for sweep, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Original sensitivity output missing: {path}. "
                "Run the original step_05 analysis on the machine holding output/."
            )
        df = pd.read_csv(path)
        require(df, ["analysis", "trial_data_name", "representation", "score",
                     "auroc_real_vs_simulation"], str(path))
        df = df.loc[
            (df["analysis"] == "m4m8_training_family_real_vs_simulation")
            & (df["representation"] == "nn1")
            & (df["score"] == "knn")
        ].copy()
        if "application_query" not in df:
            raise ValueError(f"{path}: application_query missing; cannot separate real and H1")
        df = df[df["application_query"].isin(("r1", "m9"))]
        df["sweep"] = sweep
        df["condition"] = df["knn_k"] if sweep == "k" else df["split_seed"]
        df["auroc"] = pd.to_numeric(df["auroc_real_vs_simulation"], errors="coerce")
        if df.empty:
            raise ValueError(f"No cfg3 m4m8 training-family NP rows in {path}")
        parts.append(df[["sweep", "trial_data_name", "application_query",
                         "knn_k", "split_seed", "condition", "auroc",
                         "n_real", "n_simulation"]].copy())
    full = pd.concat(parts, ignore_index=True)

    keys = ["sweep", "trial_data_name", "application_query", "condition"]
    if full.duplicated(keys).any():
        raise ValueError("Duplicated sweep rows in original outputs; inspect raw CSV.")
    expected = {"k": set(K_VALUES), "split_seed": set(SEEDS)}
    for (sweep, design, query), group in full.groupby(
            ["sweep", "trial_data_name", "application_query"]):
        observed = set(group["condition"].astype(int))
        if observed != expected[sweep]:
            raise ValueError(f"Incomplete {sweep} for {design}/{query}: {observed}")
    if set(full[full.sweep == "k"].trial_data_name) != set(DESIGNS):
        raise ValueError("Missing published design in k sensitivity")
    if set(full[full.sweep == "split_seed"].trial_data_name) != {DESIGNS[0]}:
        raise ValueError("Original split-seed sweep should cover only 380-trial design")
    write(full, "01_auroc_sensitivity_individual.csv")
    summary = full.groupby(
        ["sweep", "trial_data_name", "application_query"], as_index=False
    ).agg(n_conditions=("auroc", "size"), mean_auroc=("auroc", "mean"),
          min_auroc=("auroc", "min"), max_auroc=("auroc", "max"),
          sd_auroc=("auroc", "std"))
    write(summary, "02_auroc_sensitivity_summary.csv")
    return summary


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def make_core():
    for p in (ROOT, ROOT / "src", ROOT / "experiments" / "representation_learning",
              ROOT / "experiments" / "latent_support"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    core = module(CORE_PATH, "camera_ready_sensitivity_core")
    cfg = module(CFG_PATH, "camera_ready_sensitivity_cfg3")
    assert cfg.CONFIG_NAME == "cfg3"
    core.REPRESENTATION_TRAIN_TRIAL_DATA_NAME = "general_400"
    core.REPRESENTATION_HPARAMS = dict(cfg.REPRESENTATION_HPARAMS)
    core.MAX_POINTS_PER_DATA = 1000
    core.SEED = 31
    core.KNN_K = 10
    core.SCORE_KINDS = ("knn",)
    return core


def idx(core, exp, name, split):
    return np.asarray(core.indices_for_data_split(exp, name, split), dtype=int)


def compatibility(cal, test):
    return GLOBAL_CORE.compatibility_from_calibration(cal, test)


GLOBAL_CORE = None


def family_balance():
    global GLOBAL_CORE
    core = make_core()
    GLOBAL_CORE = core
    rep = core.load_representation("nn1", TRAIN)
    if rep is None:
        raise FileNotFoundError("cfg3 nn1 checkpoint for general_400 / m4--m8 is missing")
    results = []
    subject_rows = []
    paired_rows = []
    for design in DESIGNS:
        core.TRIAL_DATA_NAME = design
        exp = core.load_experiment_data(
            design, data_names=("r1", "m9") + TRAIN, seed=31, full_real=True
        )
        latent, _ = core.encode_representation(rep, exp)
        scale = core.fit_latent_scale(latent, exp, TRAIN)
        z = np.asarray(scale.transform(latent), dtype=float)

        neighbors = {}
        for model in TRAIN:
            reference = z[idx(core, exp, model, "train")]
            if len(reference) < 10:
                raise ValueError(f"Too few reference participants for {model}")
            neighbors[model] = NearestNeighbors(
                n_neighbors=10, metric="euclidean"
            ).fit(reference)

        def d(model, points):
            return neighbors[model].kneighbors(points, return_distance=True)[0].mean(axis=1)

        def min_dist(models, points):
            return np.min(np.column_stack([d(m, points) for m in models]), axis=1)

        query = {}
        for model in TRAIN + ("m9",):
            test = idx(core, exp, model, "test")
            if len(test) == 0:
                raise ValueError(f"No TEST records for {design}/{model}")
            query[model] = z[test]


        real = np.asarray(core.all_indices_for_data(exp, "r1"), dtype=int)
        if len(real) == 0:
            raise ValueError(f"No empirical records for {design}")
        query["r1"] = z[real]

        condition_scores = {}
        for condition, ow_models in [("full_4OW_1AW", OW)] + [
                (f"balanced_{m}_1OW_1AW", (m,)) for m in OW]:
            family_models = {"option_wise": ow_models, "attribute_wise": AW}
            calibration = {}
            for family, models in family_models.items():
                val = np.concatenate([idx(core, exp, m, "validation") for m in models])
                if not len(val):
                    raise ValueError(f"No calibration subjects for {family}")
                calibration[family] = min_dist(models, z[val])

            scores = {}
            for name, pts in query.items():
                family_c = {
                    family: compatibility(calibration[family], min_dist(models, pts))
                    for family, models in family_models.items()
                }
                c_ow, c_aw = family_c["option_wise"], family_c["attribute_wise"]
                if len(c_ow) != len(pts) or len(c_aw) != len(pts):
                    raise AssertionError("Query cardinality mismatch")
                cmax = np.maximum(c_ow, c_aw)
                pred = np.where(c_ow >= c_aw, "option_wise", "attribute_wise")
                scores[name] = cmax
                if name in TRAIN:
                    true_family = "option_wise" if name in OW else "attribute_wise"
                    accuracy = float(np.mean(pred == true_family))
                else:
                    true_family, accuracy = "", float("nan")
                results.append(dict(
                    design=design, condition=condition, query=name,
                    true_family=true_family, n=len(pts),
                    mean_cmax=float(np.mean(cmax)),
                    median_cmax=float(np.median(cmax)),
                    family_accuracy=accuracy))
                for i, value in enumerate(cmax):
                    subject_rows.append(dict(design=design, condition=condition,
                        query=name, subject_order=i, c_ow=float(c_ow[i]),
                        c_aw=float(c_aw[i]), cmax=float(value),
                        predicted_family=str(pred[i])))
            condition_scores[condition] = scores
            pool = np.concatenate([scores[m] for m in TRAIN])
            for app in ("r1", "m9"):
                auc = roc_auc_score(
                    np.r_[np.zeros(len(pool)), np.ones(len(scores[app]))],
                    np.r_[1 - pool, 1 - scores[app]])
                results.append(dict(
                    design=design, condition=condition, query=f"{app}_AUROC",
                    true_family="", n=len(scores[app]), mean_cmax=np.nan,
                    median_cmax=np.nan, family_accuracy=np.nan,
                    auroc=float(auc)))


        full_scores = condition_scores["full_4OW_1AW"]
        for ow_model in OW:
            balanced_name = f"balanced_{ow_model}_1OW_1AW"
            balanced_scores = condition_scores[balanced_name]
            for app in ("r1", "m9"):
                for source_name, source in (("full_4OW_1AW", full_scores),
                                            (balanced_name, balanced_scores)):
                    negative = np.concatenate((source[ow_model], source["m8"]))
                    positive = source[app]
                    auc = roc_auc_score(
                        np.r_[np.zeros(len(negative)), np.ones(len(positive))],
                        np.r_[1 - negative, 1 - positive])
                    paired_rows.append(dict(design=design, ow_model=ow_model,
                        support=source_name, application_query=app,
                        n_simulation=len(negative), n_application=len(positive),
                        auroc=float(auc)))
    frame = pd.DataFrame(results)
    write(frame, "03_family_balance_all_conditions.csv")
    write(pd.DataFrame(subject_rows), "04_family_balance_subjects.csv")
    paired_frame = pd.DataFrame(paired_rows)
    write(paired_frame, "06_family_balance_matched_auroc.csv")
    comparisons = []
    for (design, ow_model, app), group in paired_frame.groupby(
            ["design", "ow_model", "application_query"]):
        full_auc = float(group.loc[group.support == "full_4OW_1AW", "auroc"].iloc[0])
        balance_key = f"balanced_{ow_model}_1OW_1AW"
        balanced_auc = float(group.loc[group.support == balance_key, "auroc"].iloc[0])
        comparisons.append(dict(
            design=design, ow_model=ow_model, application_query=app,
            full_auroc=full_auc, balanced_auroc=balanced_auc,
            balanced_minus_full=balanced_auc - full_auc))
    write(pd.DataFrame(comparisons), "07_family_balance_paired_differences.csv")


    summary_rows = []
    for (design, query_name), group in frame.groupby(["design", "query"]):
        metric = ("auroc" if query_name.endswith("_AUROC")
                  else "family_accuracy" if query_name in TRAIN else "median_cmax")
        full = group.loc[group.condition == "full_4OW_1AW", metric].astype(float)
        balanced = group.loc[group.condition.str.startswith("balanced_"), metric].astype(float)
        if len(full) != 1 or len(balanced) != len(OW):
            raise ValueError(f"Missing full/balanced rows for {design}/{query_name}")
        summary_rows.append(dict(design=design, query=query_name, metric=metric,
            full_value=float(full.iloc[0]), balanced_mean=float(balanced.mean()),
            balanced_min=float(balanced.min()), balanced_max=float(balanced.max()),
            n_balanced_configs=len(balanced)))
    summary = pd.DataFrame(summary_rows)
    write(summary, "05_family_balance_comparison.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "sensitivity", "balance"),
                        default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "sensitivity"):
        print(sensitivity().to_string(index=False))
    if args.mode in ("all", "balance"):
        print(family_balance().to_string(index=False))
    (OUT / "README_results.txt").write_text(
        "Read 01-02 for k/split AUROC; 03-05 for post-hoc family-cardinality analysis.\n"
        "No NP retraining. Split-seed control uses original 380-trial-only sweep.\n"
        "Balanced OW support holds the trained encoder fixed; not a balanced-training experiment.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
