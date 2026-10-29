from __future__ import annotations

import argparse
import json
import hashlib

import numpy as np
import pandas as pd

from config import (FAMILIES, MODEL_IDS, OUT, PRIMARY_K, SCENARIOS,
                    HEURISTIC_CONTROL, scenario_dir)
from common import (auc, bootstrap_auc, checkpoint_path, compatibility,
                    encode, ensure, get_real, knn_score, load_encoder,
                    standardized, write_json)


def _load_sim(path, expected: str | None = None):
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run generate_scenarios.py")
    with np.load(path, allow_pickle=False) as f:
        y = np.asarray(f["choices"], dtype=np.float32)
        split = np.asarray(f["split"]).astype(str) if "split" in f else None
        probs = np.asarray(f["probabilities"]) if "probabilities" in f else None
    if expected and split is None:
        raise ValueError(f"{path} has no split labels")
    return y, split, probs


def _encode_cache(out_path, *, model, checkpoint, trials, choices,
                  checkpoint_identity, device, batch, overwrite):


    digest = hashlib.sha256()
    for item in (np.asarray(trials), np.asarray(choices)):
        contiguous = np.ascontiguousarray(item)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    digest.update(checkpoint_identity.encode())
    signature = digest.hexdigest()
    if out_path.exists() and not overwrite:
        with np.load(out_path, allow_pickle=False) as f:
            if "signature" in f and str(f["signature"].item()) == signature:
                arr = np.asarray(f["z"])
                if len(arr) == len(choices):
                    return arr
        print(f"[refresh stale cache] {out_path}")
    mask = np.ones(choices.shape[:2], dtype=bool)
    z = encode(model, checkpoint, trials, choices, mask, device, batch)
    ensure(out_path.parent)
    np.savez_compressed(out_path, z=z, signature=np.array(signature))
    return z


def _support_raw(query, reference_by_model, k):
    raw = {}
    for family, members in FAMILIES.items():
        distances = [knn_score(query, reference_by_model[m], k)
                     for m in members]
        raw[family] = np.min(np.column_stack(distances), axis=1)
    return raw


def _family_compat(z, refs, cal_scores, k):
    d = _support_raw(z, refs, k)
    return {name:compatibility(cal_scores[name], d[name]) for name in FAMILIES}


def analyze(scenarios=SCENARIOS, *, checkpoint=None, device="cpu",
            batch=128, k=PRIMARY_K, overwrite_cache=False, bootstrap=1000):
    if bootstrap < 10:
        raise ValueError("At least 10 bootstrap draws requested")
    real = get_real()
    split_file = OUT / "real_split.json"
    if not split_file.exists():
        raise FileNotFoundError("Run fit_models.py to establish disjoint real subject sets")
    with split_file.open(encoding="utf-8") as f:
        real_split = json.load(f)
    eval_idx = np.asarray(real_split["evaluation_indices"], dtype=int)
    cal_idx = set(real_split["calibration_indices"])
    if set(eval_idx) & cal_idx:
        raise ValueError("Leakage: real query participants appear in empirical prior")
    path = checkpoint_path(checkpoint)
    model, ckpt = load_encoder(path, device)
    stat = path.stat()
    checkpoint_identity = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    ckpt_sd = float(ckpt.get("training_config", {}).get("beta", np.nan))
    if np.isfinite(ckpt_sd) and abs(ckpt_sd - 0.1) > 1e-8:
        raise ValueError(f"Expected paper cfg3 NP beta=0.1; checkpoint beta={ckpt_sd}")
    trials = real["trials"]
    cache = ensure(OUT / "latents")
    z_real = _encode_cache(cache / "real.npz", model=model, checkpoint=ckpt,
                           trials=trials, choices=real["choices"],
                           checkpoint_identity=checkpoint_identity, device=device,
                           batch=batch, overwrite=overwrite_cache)
    y_ctrl,_,_ = _load_sim(OUT / "control" / f"{HEURISTIC_CONTROL}.npz")
    z_control = _encode_cache(cache / "control.npz", model=model, checkpoint=ckpt,
                              trials=trials, choices=y_ctrl,
                              checkpoint_identity=checkpoint_identity, device=device,
                              batch=batch, overwrite=overwrite_cache)


    embeddings = {}
    for scen in ("original",) + tuple(s for s in scenarios if s != "original"):
        per_model = {}
        for name in MODEL_IDS:
            y, split, _ = _load_sim(scenario_dir(scen) / f"{name}.npz", expected=name)
            z = _encode_cache(cache / scen / f"{name}.npz", model=model,
                              checkpoint=ckpt, trials=trials, choices=y,
                              checkpoint_identity=checkpoint_identity,
                              device=device, batch=batch, overwrite=overwrite_cache)
            per_model[name] = (z, split)
        embeddings[scen] = per_model
    original_ref = np.concatenate([embeddings["original"][name][0][
        embeddings["original"][name][1] == "reference"] for name in MODEL_IDS])
    z_mean = original_ref.mean(axis=0)
    z_sd = np.where(original_ref.std(axis=0) > 1e-6,
                    original_ref.std(axis=0), 1.0)
    z_real = standardized(z_real, z_mean, z_sd)
    z_control = standardized(z_control, z_mean, z_sd)
    write_json(OUT / "analysis_manifest.json", {
        "checkpoint":str(path), "scenario_order":list(scenarios),
        "latent_scaling":"fixed to original-prior reference simulations",
        "k":k, "calibration_formula":"(1+#cal_scores >= query_score)/(n_cal+1)",
        "calibration_unit":"per-family pooled independent validation sims",
        "real_evaluation_n":len(eval_idx), "real_calibration_n":len(cal_idx),
        "device":device, "bootstrap_repeats":bootstrap,
        "warning":"AUROC is descriptive discrimination, not a formal misspecification test",
    })
    rows = []
    summaries = []
    for s_index,scenario in enumerate(scenarios):
        per_model=embeddings[scenario]
        refs = {}
        for m in MODEL_IDS:
            z, splits=per_model[m]
            refs[m]=standardized(z[splits=="reference"], z_mean, z_sd)
            if len(refs[m]) < k:
                raise ValueError(f"Not enough reference simulations for {m}")

        cal_scores = {}
        for fam,members in FAMILIES.items():
            z_cal=np.concatenate([
                standardized(per_model[m][0][per_model[m][1]=="calibration"],
                             z_mean,z_sd) for m in members
            ])
            cal_scores[fam]=_support_raw(z_cal,refs,k)[fam]
        by_group=[]
        def add(query_z, group, model_name, subject_ids):
            comp=_family_compat(query_z,refs,cal_scores,k)
            matrix=np.column_stack([comp[f] for f in FAMILIES])
            for i in range(len(query_z)):
                entry={
                    "scenario":scenario, "group":group, "model":model_name,
                    "subject_id":str(subject_ids[i]), "cmax":float(matrix[i].max()),
                    "best_family":list(FAMILIES)[matrix[i].argmax()],
                }
                for fam in FAMILIES:
                    entry[f"c_{fam}"]=float(comp[fam][i])
                by_group.append(entry)
        for m in MODEL_IDS:
            z, splits = per_model[m]
            test=standardized(z[splits=="test"],z_mean,z_sd)
            add(test,"candidate",m,np.arange(len(test)))
        add(z_control,"h1",HEURISTIC_CONTROL,np.arange(len(z_control)))
        add(z_real[eval_idx],"real","r1",real["ids"][eval_idx])
        frame = pd.DataFrame(by_group)
        frame.to_csv(ensure(OUT / "results") / f"subjects_{scenario}.csv", index=False)
        rows.append(frame)
        c_id=frame.loc[frame.group=="candidate","cmax"].to_numpy()
        for group in ("h1","real"):
            cg=frame.loc[frame.group==group,"cmax"].to_numpy()

            au=auc(-c_id,-cg)
            ci=bootstrap_auc(-c_id,-cg,seed=3000+s_index+(group=="real"),
                             repeats=bootstrap)
            summaries.append({
                "scenario":scenario,"comparison":f"{group}_vs_candidate",
                "auc":au, "ci_low":ci[0],"ci_high":ci[1],
                "mean_candidate_cmax":float(c_id.mean()),
                "mean_query_cmax":float(cg.mean()),
                "median_query_cmax":float(np.median(cg)),
                "fraction_query_cmax_le_0_05":float(np.mean(cg<=0.05)),
                "n_candidate":len(c_id),"n_query":len(cg),
            })
        print(f"[analysis] {scenario}: "
              f"real AUROC={summaries[-1]['auc']:.3f}")
    all_subjects=pd.concat(rows,ignore_index=True)
    all_subjects.to_csv(OUT / "results" / "all_subjects.csv",index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT / "results" / "prior_sensitivity_summary.csv",index=False)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios",nargs="+",choices=SCENARIOS,default=list(SCENARIOS))
    parser.add_argument("--checkpoint",default=None)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--batch",type=int,default=128)
    parser.add_argument("--k",type=int,default=PRIMARY_K)
    parser.add_argument("--bootstrap",type=int,default=1000)
    parser.add_argument("--overwrite-cache",action="store_true")
    a=parser.parse_args()
    analyze(tuple(a.scenarios),checkpoint=a.checkpoint,device=a.device,
            batch=a.batch,k=a.k,bootstrap=a.bootstrap,
            overwrite_cache=a.overwrite_cache)


if __name__=="__main__":
    main()
