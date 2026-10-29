from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output" / "robustness" / "prior_sensitivity"

TRIAL_DESIGN = "pub_Konstantinidis2020"
TRAIN_DESIGN = "general_400"
TRAIN_MODELS = ("m4", "m5", "m6", "m7", "m8")
FAMILIES = {
    "option_wise": ("m4", "m5", "m6", "m7"),
    "attribute_wise": ("m8",),
}
HEURISTIC_CONTROL = "m9"
REAL_NAME = "r1"
NP_BETA = 0.1
TRAIN_SEED = 31
SPLIT_SEED = 31
SCENARIOS = ("original", "restricted_sd2", "restricted_sd1", "empirical")

RESTRICTED_LOGNORMAL_SDS = {"restricted_sd2": 2.0, "restricted_sd1": 1.0}
PRIMARY_K = 10
MODEL_IDS = tuple(m for members in FAMILIES.values() for m in members)

def scenario_dir(scenario: str) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    return OUT / "scenarios" / scenario
