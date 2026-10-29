# Comparing Computational Models and Intertemporal Choice Data in a Shared Latent Space

Research code for **Comparing Computational Models and Intertemporal Choice Data in a Shared Latent Space**, NeurIPS 2026 Workshop on Sim2Science: ML with Imperfect Scientific Models.

This repository contains a curated release of the intertemporal-choice experiments associated with the paper. The original research-development repository remains separate.

## Repository layout

```text
src/
  choice_generators/                 Computational choice models and priors
  trial_generators/                  Synthetic and published experimental designs
  choice_set_encoder/                Shared set-encoding components
  neural_processes_choice/           Reconstruction-based Neural Process
  neural_processes_choice_classifier/  Hybrid NP and model classifier
  neural_processes_choice_consistency/  Consistency-learning components
  neural_processes_choice_consistency_classifier/  Combined training components
  model_classifier/                  Model-supervised representation
  parameter_decoder/                 Parameter-supervised representation

experiments/
  data_generation/
    trials/                          Generate experimental trial sets
    choices/                         Simulate choice datasets
  representation_learning/           Prepare data and train latent representations
  latent_support/                    kNN compatibility, model support, and validation
    configs/                         Analysis configurations
    behavioral_summary_baseline/     Regression-summary comparison
  visualization/                     Figures and pairwise model compatibility
  design_controls/                   Trial-count and stimulus-selection control
  robustness/
    prior_sensitivity/               Prior-specification robustness
    support_sensitivity/             k and split-seed robustness; family-size control

tests/                                Selected model and data-interface checks
requirements.txt
```

## Installation

Use Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The project uses source-directory imports rather than an installed Python package. Set `PYTHONPATH=src` when running individual modules or tests, or add the absolute `src` directory to the Python import path.

## Experimental workflow

The main experiments use five training simulators (OW2–OW5 and AW2, internally `m4`–`m8`) and three published/probe conditions: two held-out known-family models, an unseen heuristic family, and real intertemporal-choice observations.

1. Generate trial sets using `experiments/data_generation/trials/run.py`.
2. Generate simulated choices using `experiments/data_generation/choices/run.py`.
3. Prepare and train representations using scripts in `experiments/representation_learning/`.
4. Analyze geometric support using `experiments/latent_support/support_core.py` and `family_support.py`.
5. Generate figures using `experiments/visualization/paper_figures.py`.
6. Run additional controls under `experiments/design_controls/` and `experiments/robustness/`.

These scripts retain research-specific configuration constants and expect the output files created by preceding stages. Several analyses require trained checkpoints and empirical datasets not distributed in this repository.

## Data and outputs

Original third-party participant-level data, trained checkpoints, and large generated outputs are **not included**. See the paper for the source publications and experimental designs. Obtain restricted or externally licensed data from their original providers and follow their applicable terms.

Generated files are written under `output/` and are excluded from version control.

## Reproducibility scope

This is a curated source-code release, not a self-contained data and checkpoint archive. Before claiming full numerical reproduction, supply the relevant external inputs, verify environment compatibility, and execute the complete pipeline. The scripts are provided for methodological transparency and reproducibility.

## Citation

Please cite the accompanying NeurIPS 2026 Sim2Science Workshop paper when using this research code.
