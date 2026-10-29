from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import (MODEL_IDS, OUT, PRIMARY_K, SCENARIOS, HEURISTIC_CONTROL,
                    SPLIT_SEED, scenario_dir)
from common import (ensure, get_real, make_model, parameter_samples, simulate,
                    write_json)

def empirical_pools(allow_nonconverged=False):
    file = OUT / "fits.csv"
    if not file.exists():
        raise FileNotFoundError("Run fit_models.py first to construct empirical priors")
    f = pd.read_csv(file)
    with (OUT / "real_split.json").open(encoding="utf-8") as stream:
        split = json.load(stream)
    cal_ids = set(split["calibration_indices"])
    ev_ids = set(split["evaluation_indices"])
    if cal_ids & ev_ids:
        raise ValueError("Calibrating and evaluating on overlapping real participants")
    f = f[f.group == "calibration"]
    if not set(f.subject_index.astype(int)).issubset(cal_ids):
        raise ValueError("Empirical priors include non-calibration participants")
    pools = {}
    for m in MODEL_IDS:
        sub = f[f.model == m]
        converged = sub[(sub.success == True) & np.isfinite(sub.fit_nll)]
        if len(converged) >= 2:
            sub = converged
        elif allow_nonconverged:

            sub = sub[np.isfinite(sub.fit_nll) & (sub.fit_nll < 1e11)]
            print(f"[PILOT ONLY] {m}: using {len(sub)} finite, "
                  "not necessarily converged fits; NOT suitable for publication.")
        else:
            sub = converged
        if len(sub) < 2:
            raise ValueError(f"Only {len(sub)} converged empirical fits for {m}. "
                             "Increase optimizer iterations/restarts or calibration N.")
        p = np.asarray([json.loads(s) for s in sub.params], dtype=float)
        if not np.isfinite(p).all() or p.shape[1] != make_model(m).n_parameters:
            raise ValueError(f"Invalid empirical parameter pool: {m}")
        pools[m] = p
    return pools


def generate(scenarios=SCENARIOS, n_reference=500, n_calibration=200, n_test=200,
             seed=SPLIT_SEED, overwrite=False, allow_nonconverged=False):
    if min(n_reference, n_calibration, n_test) < 2:
        raise ValueError("All synthetic split sizes must be >=2")
    if n_reference < PRIMARY_K:
        raise ValueError(f"At least k={PRIMARY_K} reference subjects required")
    real = get_real()
    trials = real["trials"]
    ensure(OUT)
    empirical = empirical_pools(allow_nonconverged) if "empirical" in scenarios else {}
    control = ensure(OUT / "control") / f"{HEURISTIC_CONTROL}.npz"
    if control.exists() and not overwrite:
        print(f"[reuse] {control}")
    else:
        pm = parameter_samples(HEURISTIC_CONTROL, "original", n_test, seed+990000)
        yy, pp = simulate(HEURISTIC_CONTROL, trials, pm, seed+990001)
        np.savez_compressed(control, choices=yy, probabilities=pp, parameters=pm,
                            trial_data_name=np.array("pub_Konstantinidis2020"))
    n_total = n_reference + n_calibration + n_test
    split = np.repeat(np.array(["reference", "calibration", "test"], dtype="U12"),
                      [n_reference, n_calibration, n_test])
    for scenario in scenarios:
        scenedir = ensure(scenario_dir(scenario))
        for j,name in enumerate(MODEL_IDS):
            path = scenedir / f"{name}.npz"
            if path.exists() and not overwrite:
                print(f"[reuse] {path}")
                continue


            parameter_seed = seed + 1000 * j + 13
            response_seed = seed + 1000 * j + 17
            theta = parameter_samples(name, scenario, n_total, parameter_seed,
                                      empirical.get(name))
            y, p = simulate(name, trials, theta, response_seed)
            np.savez_compressed(path, choices=y, probabilities=p,
                                parameters=theta, split=split)
            print(f"[generated] {scenario}/{name}: {n_total}")
        write_json(scenedir / "manifest.json", {
            "scenario":scenario, "seed":seed,
            "models":list(MODEL_IDS), "trial_design":"pub_Konstantinidis2020",
            "n_reference_per_model":n_reference,
            "n_calibration_per_model":n_calibration,
            "n_test_per_model":n_test,
            "parameters":("resampling vectors fitted on disjoint real calibration subjects"
                           if scenario=="empirical" else
                           "original prior" if scenario=="original" else
                           "only lognormal choice sensitivity m log-space SD restricted"),
            "encoder_retrained":False,
            "no_empirical_subject_overlap":True if scenario=="empirical" else None,
            "pilot_nonconverged_fits_allowed":bool(allow_nonconverged) if scenario=="empirical" else False,
        })
    write_json(OUT / "generation_manifest.json", {
        "scenarios":list(scenarios), "n_reference":n_reference,
        "n_calibration":n_calibration, "n_test":n_test,
        "seed":seed, "control":str(control),
    })


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--reference", type=int, default=500)
    p.add_argument("--calibration", type=int, default=200)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-nonconverged", action="store_true",
                   help="Pilot-only: allow finite but nonconverged empirical fits")
    a=p.parse_args()
    generate(tuple(a.scenarios), a.reference, a.calibration, a.test,
             a.seed, a.overwrite, a.allow_nonconverged)


if __name__ == "__main__":
    main()
