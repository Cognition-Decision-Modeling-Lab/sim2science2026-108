from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT, SCENARIOS, MODEL_IDS, scenario_dir
from common import ensure


def report():
    summary_path=OUT / "results" / "prior_sensitivity_summary.csv"
    fits_path=OUT / "fits.csv"
    if not summary_path.exists() or not fits_path.exists():
        raise FileNotFoundError("Run fit_models.py, generate_scenarios.py, and analyze_support.py first")
    summary=pd.read_csv(summary_path)
    fits=pd.read_csv(fits_path)
    ev=fits[fits.group=="evaluation"].copy()
    if ev.empty:
        raise ValueError("No held-out real-subject fit results")
    fit_table=(ev.groupby("model",as_index=False).agg(
        n_subjects=("subject_id","nunique"),
        mean_heldout_logloss=("heldout_logloss","mean"),
        mean_heldout_brier=("heldout_brier","mean"),
        mean_baseline_logloss=("heldout_baseline_logloss","mean"),
        mean_prediction_bias=("heldout_prediction_minus_observed","mean"),
        optimizer_convergence_rate=("success","mean"),
    ))
    best=(ev.sort_values(["subject_id","heldout_logloss"])
          .drop_duplicates("subject_id")
          .groupby("model").size().rename("n_best_heldout").reset_index())
    fit_table=fit_table.merge(best,on="model",how="left")
    fit_table["n_best_heldout"]=fit_table["n_best_heldout"].fillna(0).astype(int)
    dest=ensure(OUT / "results")
    fit_table.to_csv(dest / "fitted_predictive_summary.csv",index=False)


    predictive_rows=[]
    for scenario in SCENARIOS:
        for name in MODEL_IDS:
            path=scenario_dir(scenario)/f"{name}.npz"
            if not path.exists():
                continue
            with np.load(path,allow_pickle=False) as f:
                test=(f["split"].astype(str)=="test")
                p=np.asarray(f["probabilities"][test],dtype=float)
                y=np.asarray(f["choices"][test,:,0],dtype=float)
            clipped=np.clip(p,1e-10,1-1e-10)
            entropy=-clipped*np.log(clipped)-(1-clipped)*np.log1p(-clipped)
            predictive_rows.append({
                "scenario":scenario,"model":name,
                "n_simulated_subjects":len(p),
                "mean_predicted_LL_probability":float(p.mean()),
                "mean_observed_simulated_LL_choice":float(y.mean()),
                "mean_choice_entropy_nats":float(entropy.mean()),
                "fraction_near_random_p_0_45_0_55":float(np.mean((p>=0.45)&(p<=0.55))),
                "fraction_near_deterministic_p_below_0_05_or_above_0_95":float(
                    np.mean((p<=0.05)|(p>=0.95))),
            })
    predictive=pd.DataFrame(predictive_rows)
    predictive.to_csv(dest/"prior_predictive_diagnostics.csv",index=False)


    baseline=summary[summary.scenario=="original"][["comparison","auc"]].rename(
        columns={"auc":"original_auc"})
    combined=summary.merge(baseline,on="comparison",how="left")
    combined["delta_auc_vs_original"]=combined["auc"]-combined["original_auc"]
    combined.to_csv(dest / "prior_comparison.csv",index=False)

    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for i,comparison in enumerate(("real_vs_candidate","h1_vs_candidate")):
        ss=summary[summary.comparison==comparison].copy()
        labels=[s for s in SCENARIOS if s in set(ss.scenario)]
        ss=ss.set_index("scenario").loc[labels].reset_index()
        x=np.arange(len(ss))
        axs[i].errorbar(x,ss.auc,
           yerr=np.vstack([ss.auc-ss.ci_low,ss.ci_high-ss.auc]),
           fmt="o",capsize=4)
        axs[i].set_xticks(x,labels,rotation=25,ha="right")
        axs[i].set_ylim(0,1)
        axs[i].axhline(0.5,linestyle="--",alpha=0.4)
        axs[i].set_title("Real versus candidate" if i==0 else "H1 versus candidate")
        axs[i].set_ylabel("AUROC (lower compatibility = more OOD)")
    fig.tight_layout()
    fig.savefig(dest/"prior_sensitivity_auroc.png",dpi=220)
    plt.close(fig)

    def markdown_table(df):

        cols=list(df.columns)
        display=df.copy()
        for col in cols:
            if pd.api.types.is_float_dtype(display[col]):
                display[col]=display[col].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "NA")
        lines=["| "+" | ".join(cols)+" |",
               "|"+"|".join(["---"]*len(cols))+"|"]
        for row in display.itertuples(index=False,name=None):
            lines.append("| "+" | ".join(map(str,row))+" |")
        return "\n".join(lines)

    text=[
        "# Camera-ready experiment 01: prior sensitivity and fitted-model checks",
        "",
        "## Prior-predictive support",
        "",
        markdown_table(combined[[
            "scenario","comparison","auc","ci_low","ci_high",
            "mean_query_cmax","fraction_query_cmax_le_0_05",
            "delta_auc_vs_original"
        ]]),
        "",
        "## Held-out fitted-model prediction",
        "",
        markdown_table(fit_table),
        "",
        "## Prior-predictive behavioral regimes",
        "",
        markdown_table(predictive),
        "",
        "## Interpretation rules",
        "",
        "- These are calibrated prior-predictive latent-compatibility scores, not a formal composite goodness-of-fit test.",
        "- Restricted prior scenarios change only the log-scale SD of lognormal choice sensitivity 'm', when present.",
        "- The empirical scenario resamples whole parameter vectors fitted to a separate real-calibration subset.",
        "- Held-out log loss and Brier scores refer to plug-in MLE predictions on unseen trials of the real-evaluation subset.",
        "- Original-prior results here are fresh independent simulations using a frozen encoder; do not assume exact identity to paper's previously reported AUROC.",
        "- A high predictive fit relative to the subject-specific constant-choice baseline weakens claims of structural model inadequacy.",
        "- Empirically fitted priors are estimated from data and do not constitute an independent theoretically specified prior.",
        "",
    ]
    (dest/"REPORT.md").write_text("\n".join(text),encoding="utf-8")
    print("Wrote:", dest/"REPORT.md")
    print("Wrote:", dest/"prior_sensitivity_auroc.png")
    return dest


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    report()
