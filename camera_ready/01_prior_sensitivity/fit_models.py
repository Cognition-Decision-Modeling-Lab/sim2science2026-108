from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import gamma as gamma_dist, norm, truncnorm

from config import MODEL_IDS, OUT, SPLIT_SEED
from common import ensure, get_real, make_model, split_real, write_json

EPS = 1e-10


def parameter_bounds(model, q=1e-4):
    bounds, transforms = [], []
    zlo, zhi = norm.ppf([q, 1.0 - q])
    for dist, p in zip(model.prior_distribution_names,
                       model.prior_distribution_parameters, strict=True):
        if dist == "uniform":
            bounds.append((float(p[0]), float(p[1]))); transforms.append("identity")
        elif dist == "normal":
            bounds.append((p[0] + p[1] * zlo, p[0] + p[1] * zhi)); transforms.append("identity")
        elif dist == "lognormal":
            bounds.append((p[0] + p[1] * zlo, p[0] + p[1] * zhi)); transforms.append("log")
        elif dist == "gamma":
            d = gamma_dist(a=p[0], scale=p[1])
            bounds.append((np.log(max(d.ppf(q), np.exp(-20))), np.log(d.ppf(1-q))))
            transforms.append("log")
        elif dist == "truncated_normal":
            d = truncnorm((p[2]-p[0])/p[1], (p[3]-p[0])/p[1],
                          loc=p[0], scale=p[1])
            bounds.append((float(d.ppf(q)), float(d.ppf(1-q))))
            transforms.append("identity")
        else:
            raise ValueError(dist)
    if any(not np.isfinite(a) or not np.isfinite(b) or a >= b for a,b in bounds):
        raise ValueError("Invalid DE parameter bounds")
    return bounds, transforms


def natural(s, transforms):
    return np.asarray([np.exp(v) if trans == "log" else v
                       for v,trans in zip(s, transforms, strict=True)], dtype=float)


def predict(model, theta, trials):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        p = np.asarray(model._calculate_p_ll(
            trials, model.get_parameters_dict(theta)), dtype=float)
    if p.shape != (len(trials),) or not np.isfinite(p).all():
        return None
    return np.clip(p, EPS, 1-EPS)


def estimate(model, x, y, *, seed, maxiter, popsize, restarts=1):
    bounds, transforms = parameter_bounds(model)
    def obj(search):
        pp = predict(model, natural(search, transforms), x)
        if pp is None:
            return 1e12
        return float(-np.sum(y * np.log(pp) + (1-y) * np.log1p(-pp)))
    best = None
    for r in range(restarts):
        result = differential_evolution(
            obj, bounds, seed=seed + 1009*r, maxiter=maxiter,
            popsize=popsize, tol=1e-6, polish=True, workers=1,
            updating="immediate"
        )
        if best is None or result.fun < best.fun:
            best = result
    return natural(best.x, transforms), best


def fit(*, max_calibration=0, max_evaluation=0, seed=SPLIT_SEED,
        maxiter=120, popsize=8, restarts=1, holdout_fraction=0.2,
        overwrite=False):
    ensure(OUT)
    target = OUT / "fits.csv"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists. Use --overwrite to regenerate.")
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5)")
    real = get_real()
    cal, ev = split_real(len(real["ids"]), seed)
    if max_calibration > 0:
        cal = cal[:max_calibration]
    if max_evaluation > 0:
        ev = ev[:max_evaluation]
    if len(cal) < 2 or len(ev) < 2:
        raise ValueError("Need >=2 calibration and >=2 evaluation real subjects")
    write_json(OUT / "real_split.json", {
        "seed": seed, "calibration_indices": cal.tolist(),
        "evaluation_indices": ev.tolist(), "real_data_file": real["path"],
        "calibration_ids": real["ids"][cal].tolist(),
        "evaluation_ids": real["ids"][ev].tolist(),
        "holdout_fraction": holdout_fraction,
        "maxiter": maxiter, "popsize": popsize, "restarts": restarts,
    })
    rows = []

    heldouts = {}
    for i in ev:
        valid = np.flatnonzero(real["mask"][i])
        rng = np.random.default_rng(seed + 100000 + int(i))
        perm = rng.permutation(valid)
        ntest = max(1, int(round(len(valid)*holdout_fraction)))
        heldouts[int(i)] = (np.sort(perm[ntest:]), np.sort(perm[:ntest]))
    started = time.perf_counter()
    for name in MODEL_IDS:
        model = make_model(name)
        for group, indices in (("calibration", cal), ("evaluation", ev)):
            for i in indices:
                valid = np.flatnonzero(real["mask"][i])
                train, test = (valid, np.empty(0, dtype=int)) if group == "calibration" else heldouts[int(i)]
                xtrain = real["trials"][train]
                ytrain = real["choices"][i, train, 0].astype(float)
                theta, opt = estimate(
                    model, xtrain, ytrain, seed=seed + 10000*int(name[1:]) + int(i),
                    maxiter=maxiter, popsize=popsize, restarts=restarts,
                )
                ptr = predict(model, theta, xtrain)
                pte = predict(model, theta, real["trials"][test]) if len(test) else None
                yte = real["choices"][i, test, 0].astype(float) if len(test) else np.array([])
                baseline = np.clip((ytrain.sum()+1)/(len(ytrain)+2), EPS, 1-EPS)
                row = {
                    "model": name, "group": group,
                    "subject_index": int(i), "subject_id": str(real["ids"][i]),
                    "params": json.dumps(theta.tolist()), "success": bool(opt.success),
                    "nfev": int(opt.nfev), "fit_nll": float(opt.fun),
                    "train_n": len(train), "heldout_n": len(test),
                    "heldout_logloss": float(-np.mean(yte * np.log(pte) + (1-yte)*np.log1p(-pte)))
                    if pte is not None else float("nan"),
                    "heldout_brier": float(np.mean((pte-yte)**2)) if pte is not None else float("nan"),
                    "heldout_baseline_logloss": float(-np.mean(yte*np.log(baseline)
                        + (1-yte)*np.log1p(-baseline))) if len(test) else float("nan"),
                    "heldout_prediction_minus_observed": float(np.mean(pte-yte))
                    if pte is not None else float("nan"),
                }
                if ptr is None:
                    raise ValueError("Invalid best-fit probabilities")
                rows.append(row)

                pd.DataFrame(rows).to_csv(OUT / "fits_partial.csv", index=False)
                print(f"[fit] {name} {group} {len(rows)}: {row['subject_id']} "
                      f"NLL={row['fit_nll']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(target, index=False)
    (OUT / "fits_partial.csv").unlink(missing_ok=True)
    write_json(OUT / "fit_manifest.json", {
        "elapsed_seconds": time.perf_counter()-started,
        "models": list(MODEL_IDS), "n_calibration": len(cal),
        "n_evaluation": len(ev), "n_fit_rows": len(df),
        "optimizers": {"maxiter":maxiter,"popsize":popsize,"restarts":restarts},
        "note": "Plug-in maximum likelihood fits, not Bayesian posteriors."
    })
    print(f"Saved: {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calibration", type=int, default=0,
                        help="Limit calibration subjects for a quick pilot; 0=all")
    parser.add_argument("--max-evaluation", type=int, default=0,
                        help="Limit evaluation subjects for a quick pilot; 0=all")
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    fit(max_calibration=args.max_calibration,
        max_evaluation=args.max_evaluation, maxiter=args.maxiter,
        popsize=args.popsize, restarts=args.restarts, seed=args.seed,
        holdout_fraction=args.holdout_fraction, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
