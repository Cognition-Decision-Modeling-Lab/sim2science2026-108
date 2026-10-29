from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from config import (
    ROOT, OUT, NP_BETA, MODEL_IDS, TRAIN_DESIGN, TRAIN_MODELS,
    TRAIN_SEED, TRIAL_DESIGN, REAL_NAME, RESTRICTED_LOGNORMAL_SDS,
)

sys.path.insert(0, str(ROOT / "src"))
from choice_generators import (
    M4Exponential, M5DoubleExponential, M6ConstantSensitivity,
    M7AdditiveUtility, M8ProportionalDifference,
    M9PriorityLexicographicSemiorder,
)
from neural_processes_choice import ChoiceNPConfig, ChoiceNeuralProcess

FACTORIES = {
    "m4": M4Exponential,
    "m5": M5DoubleExponential,
    "m6": M6ConstantSensitivity,
    "m7": M7AdditiveUtility,
    "m8": M8ProportionalDifference,
    "m9": M9PriorityLexicographicSemiorder,
}


def make_model(name: str):
    return FACTORIES[name]()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: dict) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def get_real() -> dict:
    root = ROOT / "output" / "choices" / TRIAL_DESIGN / REAL_NAME
    paths = sorted(root.glob("n*/data.npz"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one existing real-data file under {root}; found {len(paths)}. "
            "Run the submission's step_02_generate_choices for r1 first."
        )
    with np.load(paths[0], allow_pickle=False) as f:
        x = np.asarray(f["trials"], dtype=np.float32)
        y = np.asarray(f["choices"], dtype=np.float32)
        mask = np.asarray(f["trial_mask"], dtype=bool) if "trial_mask" in f else np.ones(y.shape[:2], bool)
        ids = np.asarray(f["subject_ids"]).astype(str) if "subject_ids" in f else np.arange(y.shape[0]).astype(str)
    if x.ndim != 2 or x.shape[1] != 4 or y.ndim != 3 or y.shape[-1] != 1:
        raise ValueError("Expected shared trials (T,4) and real choices (N,T,1)")
    if x.shape[0] != y.shape[1] or mask.shape != y.shape[:2] or len(ids) != len(y):
        raise ValueError("Malformed real-data shapes")
    if len(set(ids)) != len(ids):
        raise ValueError("Real subject identifiers must be unique")
    return {"trials": x, "choices": y, "mask": mask, "ids": ids, "path": str(paths[0])}


def split_real(n: int, seed: int, calibration_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if n < 8:
        raise ValueError("At least eight real subjects required for nonoverlapping calibration/evaluation")
    if not 0.1 <= calibration_fraction <= 0.9:
        raise ValueError("calibration_fraction must be in [0.1, 0.9]")
    order = np.random.default_rng(seed).permutation(n)
    nc = int(round(n * calibration_fraction))
    return np.sort(order[:nc]), np.sort(order[nc:])


def parameter_samples(name: str, scenario: str, n: int, seed: int, empirical=None) -> np.ndarray:
    model = make_model(name)
    if scenario == "empirical":
        if empirical is None or len(empirical) < 2:
            raise ValueError(f"Insufficient independent fitted parameter rows for {name}")
        p = np.asarray(empirical, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != model.n_parameters or not np.isfinite(p).all():
            raise ValueError(f"Invalid empirical parameter pool for {name}")

        return p[np.random.default_rng(seed).integers(0, len(p), size=n)]
    if scenario in RESTRICTED_LOGNORMAL_SDS:
        model.prior_distribution_parameters = [
            list(p) for p in model.prior_distribution_parameters
        ]
        for j, (param_name, prior_name) in enumerate(
            zip(model.parameter_names, model.prior_distribution_names, strict=True)
        ):
            if param_name == "m" and prior_name == "lognormal":
                model.prior_distribution_parameters[j][1] = RESTRICTED_LOGNORMAL_SDS[scenario]
    elif scenario != "original":
        raise ValueError(f"Unknown prior scenario {scenario}")
    return model.sample_parameters(seed=seed, n_subjects=n)


def simulate(name: str, trials: np.ndarray, params: np.ndarray, seed: int):
    model = make_model(name)
    y, mask = model.generate_choices(trials, params, seed=seed)
    p = model._choice_probabilities(np.asarray(trials, dtype=np.float64), params, mask)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f"Nonfinite/invalid probabilities for {name}")
    return y.astype(np.int8), p


def checkpoint_path(override: str | None = None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    group = "_".join(TRAIN_MODELS)
    root = ROOT / "output" / "latent_models" / "neural_process" / TRAIN_DESIGN / group
    paths = sorted(root.glob(f"n*_z8/s{TRAIN_SEED}/b{NP_BETA:g}/checkpoint.pt"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one cfg3 NP checkpoint under {root}; found {len(paths)}. "
            "Supply --checkpoint PATH to select the exact submission checkpoint."
        )
    return paths[0]


def load_encoder(ckpt_path: Path, device: str = "cpu"):
    import torch
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = ChoiceNPConfig(**checkpoint["model_config"])
    model = ChoiceNeuralProcess(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    return model, checkpoint


def encode(model, checkpoint: dict, trials: np.ndarray, choices: np.ndarray,
           mask: np.ndarray, device: str = "cpu", batch: int = 128) -> np.ndarray:
    import torch
    trials = np.asarray(trials, dtype=np.float32)
    choices = np.asarray(choices, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if trials.ndim != 2 or trials.shape[0] != choices.shape[1]:
        raise ValueError("Encoder expects one common trial design")
    if choices.ndim != 3 or choices.shape[-1] != 1 or mask.shape != choices.shape[:2]:
        raise ValueError("Invalid choice/mask shapes")
    std = checkpoint.get("standardizer")
    if std is not None:
        mean = np.asarray(std["mean"], dtype=np.float32)
        scale = np.asarray(std["scale"], dtype=np.float32)
        trials = (trials - mean) / scale
    t = torch.as_tensor(trials, dtype=torch.float32, device=device)
    result = []
    for i in range(0, len(choices), batch):
        c = torch.as_tensor(choices[i:i+batch], dtype=torch.float32, device=device)
        m = torch.as_tensor(mask[i:i+batch], dtype=torch.bool, device=device)
        with torch.no_grad():
            z, _ = model.encode_context(t.unsqueeze(0).expand(len(c), -1, -1), c, m)
        result.append(z.cpu().numpy().astype(np.float64))
    if not result:
        raise ValueError("No subjects to encode")
    z = np.concatenate(result)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite latent coordinates")
    return z


def standardized(z: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(z) - mean) / sd


def knn_score(query: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    if len(ref) < k or k < 1:
        raise ValueError("Reference size must be >= k >= 1")

    chunks = []
    for i in range(0, len(query), 256):
        d = cdist(query[i:i + 256], ref, metric="euclidean")
        chunks.append(np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1))
    return np.concatenate(chunks)


def compatibility(calibration: np.ndarray, values: np.ndarray) -> np.ndarray:
    calibration = np.sort(np.asarray(calibration, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if len(calibration) < 1 or not np.isfinite(calibration).all():
        raise ValueError("Invalid calibration scores")
    if not np.isfinite(values).all():
        raise ValueError("Invalid query scores")

    return (1 + len(calibration) - np.searchsorted(calibration, values, side="left")) / (len(calibration) + 1)


def auc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    a = np.asarray(id_scores, dtype=float)
    b = np.asarray(ood_scores, dtype=float)
    if not len(a) or not len(b):
        return float("nan")
    ranks = rankdata(np.r_[a, b], method="average")
    return float((ranks[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b)))


def bootstrap_auc(a: np.ndarray, b: np.ndarray, seed: int = 31, repeats: int = 1000):
    rng = np.random.default_rng(seed)
    reps = np.asarray([auc(rng.choice(a, len(a), replace=True),
                           rng.choice(b, len(b), replace=True))
                       for _ in range(repeats)])
    return np.quantile(reps, [0.025, 0.975]).tolist()
