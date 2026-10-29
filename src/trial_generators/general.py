from numbers import Integral, Real

import numpy as np
from scipy.stats import qmc


TRIAL_COLUMNS = (
    "amount_ss",
    "delay_ss",
    "amount_ll",
    "delay_ll",
)

DEFAULT_MIN_TRIALS = 20
GENERAL_400_MAX_TRIALS = 400

DEFAULT_AMOUNT_MIN = 10.0
DEFAULT_AMOUNT_MAX = 500.0
DEFAULT_DELAY_MIN_DAYS = 0.0
DEFAULT_DELAY_MAX_DAYS = 1200.0

DEFAULT_MIN_AMOUNT_GAP = 1.0
DEFAULT_MIN_DELAY_GAP_DAYS = 1.0
DEFAULT_IMMEDIATE_SS_FRACTION = 0.50


def _validate_integer(name: str, value: int, minimum: int) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _validate_finite_real(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _ordered_pair(
    u_first: np.ndarray,
    u_second: np.ndarray,
    lower: float,
    upper: float,
    min_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    value_first = lower + u_first * (upper - lower)
    value_second = lower + u_second * (upper - lower)

    pair_low = np.minimum(value_first, value_second)
    pair_high = np.maximum(value_first, value_second)


    too_close = (pair_high - pair_low) < min_gap
    if np.any(too_close):
        pair_high[too_close] = np.minimum(
            upper,
            pair_low[too_close] + min_gap,
        )
        still_too_close = (pair_high - pair_low) < min_gap
        pair_low[still_too_close] = np.maximum(
            lower,
            pair_high[still_too_close] - min_gap,
        )

    return pair_low, pair_high


def _generate_one_participant(
    n_trials: int,
    rng: np.random.Generator,
    amount_min: float,
    amount_max: float,
    delay_min_days: float,
    delay_max_days: float,
    min_amount_gap: float,
    min_delay_gap_days: float,
    immediate_ss_fraction: float,
) -> np.ndarray:


    lhs = qmc.LatinHypercube(
        d=4,
        scramble=True,
        seed=rng,
    )
    unit = lhs.random(n_trials)

    amount_ss, amount_ll = _ordered_pair(
        unit[:, 0],
        unit[:, 1],
        amount_min,
        amount_max,
        min_amount_gap,
    )

    delay_ss, delay_ll = _ordered_pair(
        unit[:, 2],
        unit[:, 3],
        delay_min_days,
        delay_max_days,
        min_delay_gap_days,
    )


    n_immediate = int(round(n_trials * immediate_ss_fraction))
    if n_immediate > 0:
        immediate_indices = rng.choice(
            n_trials,
            size=n_immediate,
            replace=False,
        )
        delay_ss[immediate_indices] = 0.0


        immediate_ll = (
            delay_min_days
            + unit[immediate_indices, 3]
            * (delay_max_days - delay_min_days)
        )
        immediate_ll = np.maximum(
            immediate_ll,
            min_delay_gap_days,
        )
        delay_ll[immediate_indices] = np.minimum(
            immediate_ll,
            delay_max_days,
        )

    trials = np.column_stack(
        [amount_ss, delay_ss, amount_ll, delay_ll]
    ).astype(np.float32)

    if np.any(trials[:, 0] >= trials[:, 2]):
        raise RuntimeError("Generated trial violates amount_ss < amount_ll.")
    if np.any(trials[:, 1] >= trials[:, 3]):
        raise RuntimeError("Generated trial violates delay_ss < delay_ll.")
    if np.any(~np.isfinite(trials)):
        raise RuntimeError("Generated non-finite trial values.")


    return trials[rng.permutation(n_trials)]


def _padded_participant_trials(
    n_subjects: int,
    min_trials: int,
    max_trials: int,
    amount_min: float = DEFAULT_AMOUNT_MIN,
    amount_max: float = DEFAULT_AMOUNT_MAX,
    delay_min_days: float = DEFAULT_DELAY_MIN_DAYS,
    delay_max_days: float = DEFAULT_DELAY_MAX_DAYS,
    min_amount_gap: float = DEFAULT_MIN_AMOUNT_GAP,
    min_delay_gap_days: float = DEFAULT_MIN_DELAY_GAP_DAYS,
    immediate_ss_fraction: float = DEFAULT_IMMEDIATE_SS_FRACTION,
    seed: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_integer("n_subjects", n_subjects, 1)
    _validate_integer("min_trials", min_trials, 2)
    _validate_integer("max_trials", max_trials, min_trials)
    _validate_integer("seed", seed, 0)

    for name, value in (
        ("amount_min", amount_min),
        ("amount_max", amount_max),
        ("delay_min_days", delay_min_days),
        ("delay_max_days", delay_max_days),
        ("min_amount_gap", min_amount_gap),
        ("min_delay_gap_days", min_delay_gap_days),
        ("immediate_ss_fraction", immediate_ss_fraction),
    ):
        _validate_finite_real(name, value)

    if amount_min < 0.0:
        raise ValueError("amount_min must be >= 0.")
    if amount_max <= amount_min:
        raise ValueError("amount_max must be > amount_min.")
    if delay_min_days < 0.0:
        raise ValueError("delay_min_days must be >= 0.")
    if delay_max_days <= delay_min_days:
        raise ValueError("delay_max_days must be > delay_min_days.")
    if min_amount_gap <= 0.0:
        raise ValueError("min_amount_gap must be > 0.")
    if min_delay_gap_days <= 0.0:
        raise ValueError("min_delay_gap_days must be > 0.")
    if min_amount_gap >= (amount_max - amount_min):
        raise ValueError("min_amount_gap is too large for the amount range.")
    if min_delay_gap_days >= (delay_max_days - delay_min_days):
        raise ValueError("min_delay_gap_days is too large for the delay range.")
    if not 0.0 <= immediate_ss_fraction <= 1.0:
        raise ValueError("immediate_ss_fraction must be in [0, 1].")

    rng = np.random.default_rng(seed)
    n_trials_per_subject = rng.integers(
        min_trials,
        max_trials + 1,
        size=n_subjects,
        dtype=np.int32,
    )

    trials = np.zeros(
        (n_subjects, max_trials, len(TRIAL_COLUMNS)),
        dtype=np.float32,
    )
    trial_mask = np.zeros(
        (n_subjects, max_trials),
        dtype=bool,
    )

    for subject_index, n_trials in enumerate(n_trials_per_subject):
        participant_trials = _generate_one_participant(
            n_trials=int(n_trials),
            rng=rng,
            amount_min=float(amount_min),
            amount_max=float(amount_max),
            delay_min_days=float(delay_min_days),
            delay_max_days=float(delay_max_days),
            min_amount_gap=float(min_amount_gap),
            min_delay_gap_days=float(min_delay_gap_days),
            immediate_ss_fraction=float(immediate_ss_fraction),
        )
        trials[subject_index, :n_trials] = participant_trials
        trial_mask[subject_index, :n_trials] = True

    return trials, trial_mask, n_trials_per_subject


def general_400(
    n_subjects: int,
    min_trials: int = DEFAULT_MIN_TRIALS,
    amount_min: float = DEFAULT_AMOUNT_MIN,
    amount_max: float = DEFAULT_AMOUNT_MAX,
    delay_min_days: float = DEFAULT_DELAY_MIN_DAYS,
    delay_max_days: float = DEFAULT_DELAY_MAX_DAYS,
    min_amount_gap: float = DEFAULT_MIN_AMOUNT_GAP,
    min_delay_gap_days: float = DEFAULT_MIN_DELAY_GAP_DAYS,
    immediate_ss_fraction: float = DEFAULT_IMMEDIATE_SS_FRACTION,
    seed: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _padded_participant_trials(
        n_subjects=n_subjects,
        min_trials=min_trials,
        max_trials=GENERAL_400_MAX_TRIALS,
        amount_min=amount_min,
        amount_max=amount_max,
        delay_min_days=delay_min_days,
        delay_max_days=delay_max_days,
        min_amount_gap=min_amount_gap,
        min_delay_gap_days=min_delay_gap_days,
        immediate_ss_fraction=immediate_ss_fraction,
        seed=seed,
    )
