import numpy as np
import pandas as pd


DAYS_PER_MONTH = np.float32(365.0 / 12.0)


def konstantinidis_2020_exp1_intertemporal() -> pd.DataFrame:
    amount_ll_levels = np.arange(
        120.0,
        501.0,
        20.0,
        dtype=np.float32,
    )
    delay_ll_levels = (
        np.arange(
            2.0,
            39.0,
            2.0,
            dtype=np.float32,
        )
        * DAYS_PER_MONTH
    )

    trials = [
        (100.0, 0.0, amount_ll, delay_ll)
        for amount_ll in amount_ll_levels
        for delay_ll in delay_ll_levels
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


def thrailkill_2023_delay_discounting() -> pd.DataFrame:
    trials = [
        (54.0, 0.0, 55.0, 117.0),
        (55.0, 0.0, 75.0, 61.0),
        (19.0, 0.0, 25.0, 53.0),
        (31.0, 0.0, 85.0, 7.0),
        (14.0, 0.0, 25.0, 19.0),
        (47.0, 0.0, 50.0, 160.0),
        (15.0, 0.0, 35.0, 13.0),
        (25.0, 0.0, 60.0, 14.0),
        (78.0, 0.0, 80.0, 162.0),
        (40.0, 0.0, 55.0, 62.0),
        (11.0, 0.0, 30.0, 7.0),
        (67.0, 0.0, 75.0, 119.0),
        (34.0, 0.0, 35.0, 186.0),
        (27.0, 0.0, 50.0, 21.0),
        (69.0, 0.0, 85.0, 91.0),
        (49.0, 0.0, 60.0, 89.0),
        (80.0, 0.0, 85.0, 157.0),
        (24.0, 0.0, 35.0, 29.0),
        (33.0, 0.0, 80.0, 14.0),
        (28.0, 0.0, 30.0, 179.0),
        (34.0, 0.0, 50.0, 30.0),
        (25.0, 0.0, 30.0, 80.0),
        (41.0, 0.0, 75.0, 20.0),
        (54.0, 0.0, 60.0, 111.0),
        (54.0, 0.0, 80.0, 30.0),
        (22.0, 0.0, 25.0, 136.0),
        (20.0, 0.0, 55.0, 7.0),
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
