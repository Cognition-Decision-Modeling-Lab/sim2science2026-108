HEURISTIC_AMOUNT_GAP_SCALE = 490.0
HEURISTIC_DELAY_GAP_SCALE = 1200.0


CHOICE_MODEL_CONFIGS = {
    "m1": {
        "parameter_names": ["k", "m"],
        "parameter_descriptions": [
            "discount rate",
            "choice sensitivity",
        ],
        "prior_names": [
            "lognormal",
            "lognormal",
        ],
        "prior_parameters": [
            [-5.0, 3.0],
            [1.0, 5.0],
        ],
    },
    "m2": {
        "parameter_names": ["k", "s", "m"],
        "parameter_descriptions": [
            "discount rate",
            "time sensitivity",
            "choice sensitivity",
        ],
        "prior_names": [
            "lognormal",
            "truncated_normal",
            "lognormal",
        ],
        "prior_parameters": [
            [-5.0, 3.0],
            [0.8, 0.3, 0.0, float("inf")],
            [1.0, 5.0],
        ],
    },
    "m3": {
        "parameter_names": ["u", "v", "w"],
        "parameter_descriptions": [
            "payoff sensitivity",
            "delay sensitivity",
            "payoff attention weight",
        ],
        "prior_names": [
            "truncated_normal",
            "truncated_normal",
            "uniform",
        ],
        "prior_parameters": [
            [0.8, 0.3, 0.0, float("inf")],
            [0.8, 0.3, 0.0, float("inf")],
            [0.0, 1.0],
        ],
    },
    "m4": {
        "parameter_names": ["k", "m"],
        "parameter_descriptions": [
            "constant exponential discount rate",
            "choice sensitivity",
        ],
        "prior_names": [
            "lognormal",
            "lognormal",
        ],
        "prior_parameters": [
            [-5.0, 3.0],
            [1.0, 5.0],
        ],
    },
    "m5": {
        "parameter_names": ["beta_gap", "delta", "omega", "m"],
        "parameter_descriptions": [
            "gap between fast and slow discount rates",
            "slow exponential discount rate",
            "weight on the fast exponential component",
            "choice sensitivity",
        ],
        "prior_names": [
            "lognormal",
            "lognormal",
            "uniform",
            "lognormal",
        ],
        "prior_parameters": [
            [-5.0, 3.0],
            [-5.0, 3.0],
            [0.0, 1.0],
            [1.0, 5.0],
        ],
    },
    "m6": {
        "parameter_names": ["alpha", "beta", "m"],
        "parameter_descriptions": [
            "discount scale",
            "subjective-time sensitivity",
            "choice sensitivity",
        ],
        "prior_names": [
            "lognormal",
            "truncated_normal",
            "lognormal",
        ],
        "prior_parameters": [
            [-5.0, 3.0],
            [0.8, 0.3, 0.0, float("inf")],
            [1.0, 5.0],
        ],
    },
    "m7": {
        "parameter_names": ["alpha", "beta", "lambda", "m"],
        "parameter_descriptions": [
            "payoff utility curvature",
            "delay sensitivity",
            "delay-cost weight",
            "choice sensitivity",
        ],
        "prior_names": [
            "uniform",
            "uniform",
            "gamma",
            "lognormal",
        ],
        "prior_parameters": [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.5],
            [1.0, 5.0],
        ],
    },
    "m8": {
        "parameter_names": ["delta", "sigma"],
        "parameter_descriptions": [
            "SS-versus-LL decision threshold",
            "choice sensitivity",
        ],
        "prior_names": [
            "normal",
            "gamma",
        ],
        "prior_parameters": [
            [0.0, 5.0],
            [1.0, 0.5],
        ],
    },
    "m9": {
        "parameter_names": [
            "mu_amount",
            "mu_delay",
            "sigma_threshold",
            "q_amount_first",
        ],
        "parameter_descriptions": [
            "mean amount discrimination threshold",
            "mean delay discrimination threshold",
            "trial-to-trial discrimination-threshold variability",
            "probability of inspecting amount before delay",
        ],
        "prior_names": [
            "truncated_normal",
            "truncated_normal",
            "truncated_normal",
            "uniform",
        ],
        "prior_parameters": [
            [0.35, 0.15, 0.05, 0.75],
            [0.35, 0.15, 0.05, 0.75],
            [0.15, 0.06, 0.05, 0.30],
            [0.20, 0.80],
        ],
    },
    "m10": {
        "parameter_names": [
            "theta_amount_diff",
            "theta_amount_ratio",
            "theta_delay_diff",
            "theta_delay_ratio",
            "tau",
        ],
        "parameter_descriptions": [
            "amount absolute-difference similarity threshold",
            "amount ratio similarity threshold",
            "delay absolute-difference similarity threshold",
            "delay ratio similarity threshold",
            "similarity-judgment softness",
        ],
        "prior_names": [
            "truncated_normal",
            "truncated_normal",
            "truncated_normal",
            "truncated_normal",
            "truncated_normal",
        ],
        "prior_parameters": [
            [0.15, 0.08, 0.03, 0.35],
            [0.80, 0.08, 0.55, 0.95],
            [0.15, 0.08, 0.03, 0.35],
            [0.80, 0.08, 0.55, 0.95],
            [0.07, 0.025, 0.03, 0.15],
        ],
    },
    "m11": {
        "parameter_names": ["p"],
        "parameter_descriptions": [
            "constant LL choice probability",
        ],
        "prior_names": ["uniform"],
        "prior_parameters": [[0.3, 0.7]],
    },
}
