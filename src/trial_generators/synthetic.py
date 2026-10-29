from numbers import Integral, Real

import numpy as np
import pandas as pd


DEFAULT_DELAY_MAX_DAYS = 365.0 * 38.0 / 12.0


def grid_ssll(
    n_amount_levels: int = 6,
    n_delay_levels: int = 6,
    amount_min: float = 100.0,
    amount_max: float = 500.0,
    delay_max: float = DEFAULT_DELAY_MAX_DAYS,
) -> pd.DataFrame:
    if (
        not isinstance(n_amount_levels, Integral)
        or isinstance(n_amount_levels, bool)
    ):
        raise TypeError("n_amount_levels must be an integer.")
    if n_amount_levels < 2:
        raise ValueError("n_amount_levels must be >= 2.")

    if (
        not isinstance(n_delay_levels, Integral)
        or isinstance(n_delay_levels, bool)
    ):
        raise TypeError("n_delay_levels must be an integer.")
    if n_delay_levels < 2:
        raise ValueError("n_delay_levels must be >= 2.")

    if not isinstance(amount_min, Real) or isinstance(amount_min, bool):
        raise TypeError("amount_min must be a real number.")
    if not np.isfinite(amount_min) or amount_min <= 0:
        raise ValueError("amount_min must be finite and > 0.")

    if not isinstance(amount_max, Real) or isinstance(amount_max, bool):
        raise TypeError("amount_max must be a real number.")
    if not np.isfinite(amount_max) or amount_max <= amount_min:
        raise ValueError("amount_max must be finite and > amount_min.")

    if not isinstance(delay_max, Real) or isinstance(delay_max, bool):
        raise TypeError("delay_max must be a real number.")
    if not np.isfinite(delay_max) or delay_max <= 0:
        raise ValueError("delay_max must be finite and > 0.")

    amount_levels = np.linspace(
        amount_min,
        amount_max,
        n_amount_levels,
        dtype=np.float32,
    )
    delay_levels = np.linspace(
        0.0,
        delay_max,
        n_delay_levels,
        dtype=np.float32,
    )

    amount_pairs = [
        (amount_ss, amount_ll)
        for amount_ss in amount_levels
        for amount_ll in amount_levels
        if amount_ss < amount_ll
    ]
    delay_pairs = [
        (delay_ss, delay_ll)
        for delay_ss in delay_levels
        for delay_ll in delay_levels
        if delay_ss < delay_ll
    ]

    trials = [
        (amount_ss, delay_ss, amount_ll, delay_ll)
        for amount_ss, amount_ll in amount_pairs
        for delay_ss, delay_ll in delay_pairs
    ]

    return pd.DataFrame(
        trials,
        columns=[
            "amount_ss",
            "delay_ss",
            "amount_ll",
            "delay_ll",
        ],
        dtype=np.float32,
    )
