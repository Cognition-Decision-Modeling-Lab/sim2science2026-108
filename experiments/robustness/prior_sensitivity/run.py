from __future__ import annotations

import argparse

from config import SCENARIOS
from fit_models import fit
from generate_scenarios import generate
from analyze_support import analyze
from report import report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("fit","generate","analyze","report","all"),
                   default="all")
    p.add_argument("--pilot",action="store_true",
                   help="Use a small diagnostic run. Not paper-quality inference.")
    p.add_argument("--overwrite",action="store_true",
                   help="Regenerate data/fits and refresh latent caches.")
    p.add_argument("--checkpoint",default=None,
                   help="Exact cfg3 NP b0.1 checkpoint; otherwise autodetect")
    p.add_argument("--device",default="cpu",choices=("cpu","cuda"))
    p.add_argument("--scenarios",nargs="+",choices=SCENARIOS,
                   default=list(SCENARIOS))
    p.add_argument("--maxiter",type=int,default=None)
    p.add_argument("--popsize",type=int,default=None)
    p.add_argument("--restarts",type=int,default=1)
    p.add_argument("--reference",type=int,default=None)
    p.add_argument("--calibration",type=int,default=None)
    p.add_argument("--test",type=int,default=None)
    p.add_argument("--bootstrap",type=int,default=None)
    a=p.parse_args()
    scenarios=tuple(a.scenarios)
    nref=a.reference or (40 if a.pilot else 500)
    ncal=a.calibration or (20 if a.pilot else 200)
    ntest=a.test or (20 if a.pilot else 200)
    iterations=a.maxiter or (12 if a.pilot else 120)
    popsize=a.popsize or (5 if a.pilot else 8)
    nboot=a.bootstrap or (100 if a.pilot else 1000)
    if a.stage in ("fit","all"):
        fit(max_calibration=5 if a.pilot else 0,
            max_evaluation=5 if a.pilot else 0,
            maxiter=iterations,popsize=popsize,
            restarts=a.restarts,overwrite=a.overwrite)
    if a.stage in ("generate","all"):
        generate(scenarios,n_reference=nref,n_calibration=ncal,
                 n_test=ntest,overwrite=a.overwrite,
                 allow_nonconverged=a.pilot)
    if a.stage in ("analyze","all"):
        analyze(scenarios,checkpoint=a.checkpoint,device=a.device,
                overwrite_cache=a.overwrite,bootstrap=nboot)
    if a.stage in ("report","all"):
        report()


if __name__=="__main__":
    main()
