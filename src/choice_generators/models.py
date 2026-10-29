import numpy as np
import pandas as pd
from scipy.special import expit, ndtr
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm, truncnorm

from .config import (
    CHOICE_MODEL_CONFIGS,
    HEURISTIC_AMOUNT_GAP_SCALE,
    HEURISTIC_DELAY_GAP_SCALE,
)


class BaseChoiceModel:

    def __init__(self, model_name):
        config = CHOICE_MODEL_CONFIGS[model_name]
        self.parameter_names = config["parameter_names"]
        self.parameter_descriptions = config["parameter_descriptions"]
        self.prior_distribution_names = config["prior_names"]
        self.prior_distribution_parameters = config["prior_parameters"]
        self.n_parameters = len(self.parameter_names)
        self.parameter_values = None

    def sample_parameters(
        self,
        seed=31,
        n_subjects=1,
    ):
        rng = np.random.default_rng(seed)
        parameter_values = []
        for distribution, prior in zip(
            self.prior_distribution_names,
            self.prior_distribution_parameters,
            strict=True,
        ):
            if distribution == "uniform":
                values = rng.uniform(
                    prior[0],
                    prior[1],
                    size=n_subjects,
                )
            elif distribution == "normal":
                values = rng.normal(
                    prior[0],
                    prior[1],
                    size=n_subjects,
                )
            elif distribution == "lognormal":
                values = rng.lognormal(
                    prior[0],
                    prior[1],
                    size=n_subjects,
                )
            elif distribution == "gamma":
                values = rng.gamma(
                    shape=prior[0],
                    scale=prior[1],
                    size=n_subjects,
                )
            elif distribution == "truncated_normal":
                mean, sd, lower, upper = prior
                values = truncnorm.rvs(
                    (lower - mean) / sd,
                    (upper - mean) / sd,
                    loc=mean,
                    scale=sd,
                    size=n_subjects,
                    random_state=rng,
                )
            else:
                raise ValueError(f"Unknown prior distribution: {distribution}")

            parameter_values.append(values)

        self.parameter_values = np.column_stack(parameter_values)
        return self.parameter_values

    def get_parameters_dict(self, parameter_values):
        parameter_values = np.asarray(parameter_values, dtype=np.float64)
        if parameter_values.ndim == 2 and parameter_values.shape[0] == 1:
            parameter_values = parameter_values[0]
        if parameter_values.shape != (self.n_parameters,):
            raise ValueError("Pass one row of parameter values.")

        return dict(
            zip(self.parameter_names, parameter_values, strict=True)
        )

    def plot_priors(self, bins=30, show=True):
        import matplotlib.pyplot as plt

        if self.parameter_values is None:
            raise RuntimeError("sample_parameters() must be called first.")

        figure, axes = plt.subplots(
            1,
            self.n_parameters,
            figsize=(5 * self.n_parameters, 4),
            squeeze=False,
        )

        for index, name in enumerate(self.parameter_names):
            axis = axes[0, index]
            distribution = self.prior_distribution_names[index]
            prior = self.prior_distribution_parameters[index]
            sampled_values = self.parameter_values[:, index]
            histogram_values = sampled_values
            x_label = "Parameter value"

            if distribution == "uniform":
                x = np.linspace(prior[0], prior[1], 500)
                density = np.full_like(x, 1.0 / (prior[1] - prior[0]))
                histogram_bins = bins
            elif distribution == "normal":
                x = np.linspace(
                    prior[0] - 4.0 * prior[1],
                    prior[0] + 4.0 * prior[1],
                    500,
                )
                density = norm.pdf(x, loc=prior[0], scale=prior[1])
                histogram_bins = bins
            elif distribution == "lognormal":
                prior_distribution = lognorm(
                    s=prior[1],
                    scale=np.exp(prior[0]),
                )
                lower = np.quantile(sampled_values, 0.01)
                upper = np.quantile(sampled_values, 0.90)
                if lower <= 0.0:
                    lower = np.nextafter(0.0, 1.0)
                if upper <= lower:
                    upper = lower * 10.0
                x = np.geomspace(lower, upper, 500)
                density = prior_distribution.pdf(x)
                histogram_bins = np.geomspace(
                    lower,
                    upper,
                    bins + 1,
                )
                axis.set_xscale("log")
                axis.set_xlim(lower, upper)
            elif distribution == "gamma":
                shape, scale = prior
                prior_distribution = gamma_dist(
                    a=shape,
                    scale=scale,
                )
                upper = prior_distribution.ppf(0.999)
                x = np.linspace(0.0, upper, 500)
                density = prior_distribution.pdf(x)
                histogram_bins = bins
            elif distribution == "truncated_normal":
                mean, sd, lower, upper = prior
                prior_distribution = truncnorm(
                    (lower - mean) / sd,
                    (upper - mean) / sd,
                    loc=mean,
                    scale=sd,
                )
                x = np.linspace(
                    prior_distribution.ppf(0.001),
                    prior_distribution.ppf(0.999),
                    500,
                )
                density = prior_distribution.pdf(x)
                histogram_bins = bins
            else:
                raise ValueError(f"Unknown prior distribution: {distribution}")

            axis.hist(
                histogram_values,
                bins=histogram_bins,
                density=True,
                alpha=0.4,
                label="Sampled values",
            )
            axis.plot(x, density, linewidth=2, label="Prior density")
            axis.set_title(name)
            axis.set_xlabel(x_label)
            axis.set_ylabel("Density")
            axis.legend()

        figure.tight_layout()
        if show:
            plt.show()

        return figure, axes

    def generate_choices(
        self,
        trials,
        parameter_values,
        seed=31,
        trial_mask=None,
    ):
        rng = np.random.default_rng(seed)

        parameters = np.asarray(parameter_values, dtype=np.float64)
        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, :]
        if (
            parameters.ndim != 2
            or parameters.shape[1] != self.n_parameters
        ):
            raise ValueError(
                "parameter_values must have shape "
                "(n_subjects, n_parameters)."
            )

        values, trial_mask = self._prepare_trial_values(
            trials,
            n_subjects=parameters.shape[0],
            trial_mask=trial_mask,
        )
        p_ll = self._choice_probabilities(values, parameters, trial_mask)
        if np.any(~np.isfinite(p_ll[trial_mask])):
            raise FloatingPointError(
                "The model produced non-finite choice probabilities."
            )
        if np.any((p_ll[trial_mask] < 0.0) | (p_ll[trial_mask] > 1.0)):
            raise FloatingPointError(
                "The model produced choice probabilities outside [0, 1]."
            )
        sampling_p = np.where(trial_mask, p_ll, 0.0)
        choices = rng.binomial(1, sampling_p).astype(np.int8)
        choices = choices[:, :, np.newaxis]
        choices[~trial_mask] = 0
        return choices, trial_mask

    @staticmethod
    def _prepare_trial_values(trials, n_subjects, trial_mask):
        columns = ["amount_ss", "delay_ss", "amount_ll", "delay_ll"]
        if isinstance(trials, pd.DataFrame):
            values = trials[columns].to_numpy(dtype=np.float64)
        else:
            values = np.asarray(trials, dtype=np.float64)

        if values.ndim == 2:
            if values.shape[1] != 4:
                raise ValueError("Shared trials must have shape (T, 4).")
            n_trials = values.shape[0]
            if trial_mask is None:
                trial_mask = np.ones((n_subjects, n_trials), dtype=bool)
            else:
                trial_mask = np.asarray(trial_mask, dtype=bool)
            if trial_mask.shape != (n_subjects, n_trials):
                raise ValueError(
                    "trial_mask must have shape (n_subjects, n_trials)."
                )
            BaseChoiceModel._validate_trial_values(
                values[trial_mask.any(axis=0)]
            )
            return values, trial_mask

        if values.ndim != 3:
            raise ValueError("trials must have shape (T, 4) or (S, T, 4).")
        if values.shape[0] != n_subjects or values.shape[2] != 4:
            raise ValueError(
                "Subject-specific trials must have shape "
                "(n_subjects, n_trials, 4)."
            )
        if trial_mask is None:
            raise ValueError(
                "trial_mask is required for subject-specific trials."
            )
        trial_mask = np.asarray(trial_mask, dtype=bool)
        if trial_mask.shape != values.shape[:2]:
            raise ValueError(
                "trial_mask must have shape (n_subjects, n_trials)."
            )
        if not trial_mask.any(axis=1).all():
            raise ValueError("Every subject must have at least one valid trial.")
        BaseChoiceModel._validate_trial_values(values[trial_mask])
        return values, trial_mask

    @staticmethod
    def _validate_trial_values(values):
        if values.size == 0:
            raise ValueError("trials must contain at least one valid row.")
        amount_ss, delay_ss, amount_ll, delay_ll = values.T
        if np.any(~np.isfinite(values)):
            raise ValueError("trials must contain only finite values.")
        if np.any(values < 0.0):
            raise ValueError("trial amounts and delays must be nonnegative.")
        if np.any(amount_ss >= amount_ll) or np.any(delay_ss >= delay_ll):
            raise ValueError("trials must follow the SS-LL ordering.")

    def _choice_probabilities(self, values, parameters, trial_mask):
        n_subjects, n_trials = trial_mask.shape
        p_ll = np.zeros((n_subjects, n_trials), dtype=np.float64)
        if values.ndim == 2:
            for subject_index, parameter_row in enumerate(parameters):
                valid = trial_mask[subject_index]
                if valid.all():
                    subject_trials = values
                else:
                    subject_trials = values[valid]
                p_ll[subject_index, valid] = self._calculate_p_ll(
                    subject_trials,
                    self.get_parameters_dict(parameter_row),
                )
            return p_ll

        for subject_index, parameter_row in enumerate(parameters):
            valid = trial_mask[subject_index]
            p_ll[subject_index, valid] = self._calculate_p_ll(
                values[subject_index, valid],
                self.get_parameters_dict(parameter_row),
            )
        return p_ll

    def plot_choice_proportions(self, choices, show=True, trial_mask=None):
        import matplotlib.pyplot as plt

        choices = np.asarray(choices)
        if trial_mask is None:
            choice_proportions = choices[:, :, 0].mean(axis=1)
        else:
            trial_mask = np.asarray(trial_mask, dtype=bool)
            n_valid = trial_mask.sum(axis=1)
            if np.any(n_valid == 0):
                raise ValueError("Every subject must have a valid trial.")
            choice_proportions = (
                (choices[:, :, 0] * trial_mask).sum(axis=1) / n_valid
            )
        choice_proportions = np.sort(choice_proportions)
        subject_numbers = np.arange(1, len(choice_proportions) + 1)

        figure, axis = plt.subplots(figsize=(8, 4))
        axis.scatter(subject_numbers, choice_proportions)
        axis.axhline(
            choice_proportions.mean(),
            color="tab:red",
            linestyle="--",
            label="Mean",
        )
        axis.set_xlabel("Subjects sorted by LL proportion")
        axis.set_ylabel("Proportion choosing LL")
        axis.set_ylim(0.0, 1.0)
        axis.set_title("LL choice proportions")
        axis.legend()
        figure.tight_layout()

        if show:
            plt.show()

        return figure, axis

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        raise NotImplementedError


class M1Hyperbolic(BaseChoiceModel):

    def __init__(self):
        super().__init__("m1")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        amount_ss, delay_ss, amount_ll, delay_ll = trials.T
        k = parameters["k"]
        m = parameters["m"]

        value_ss = amount_ss / (1.0 + k * delay_ss)
        value_ll = amount_ll / (1.0 + k * delay_ll)
        return expit(m * (value_ll - value_ss))


class M2Hyperboloid(BaseChoiceModel):

    def __init__(self):
        super().__init__("m2")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        amount_ss, delay_ss, amount_ll, delay_ll = trials.T
        k = parameters["k"]
        s = parameters["s"]
        m = parameters["m"]

        value_ss = amount_ss / (1.0 + (k * delay_ss) ** s)
        value_ll = amount_ll / (1.0 + (k * delay_ll) ** s)
        return expit(m * (value_ll - value_ss))


class M3DirectDifference(BaseChoiceModel):

    def __init__(self):
        super().__init__("m3")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        amount_ss, delay_ss, amount_ll, delay_ll = trials.T
        u = parameters["u"]
        v = parameters["v"]
        w = parameters["w"]

        amount_difference = amount_ll**u - amount_ss**u
        delay_difference = delay_ll**v - delay_ss**v
        d_ll = (
            w * amount_difference
            - (1.0 - w) * delay_difference
        )

        variance = (
            w * amount_difference**2
            + (1.0 - w) * delay_difference**2
            - d_ll**2
        )
        sigma = np.sqrt(np.maximum(variance, 0.0))

        p_ll = np.empty_like(d_ll)
        noisy = sigma > 1e-12
        p_ll[noisy] = ndtr(d_ll[noisy] / sigma[noisy])
        p_ll[~noisy & (d_ll > 0)] = 1.0
        p_ll[~noisy & (d_ll < 0)] = 0.0
        p_ll[~noisy & (d_ll == 0)] = 0.5
        return p_ll


class M4Exponential(BaseChoiceModel):

    def __init__(self):
        super().__init__("m4")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        amount_ss, delay_ss, amount_ll, delay_ll = trials.T
        k = parameters["k"]
        m = parameters["m"]

        value_ss = amount_ss * np.exp(-k * delay_ss)
        value_ll = amount_ll * np.exp(-k * delay_ll)
        return expit(m * (value_ll - value_ss))


class M5DoubleExponential(BaseChoiceModel):

    def __init__(self):
        super().__init__("m5")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        amount_ss, delay_ss, amount_ll, delay_ll = trials.T
        beta = parameters["delta"] + parameters["beta_gap"]
        delta = parameters["delta"]
        omega = parameters["omega"]
        m = parameters["m"]

        discount_ss = (
            omega * np.exp(-beta * delay_ss)
            + (1.0 - omega) * np.exp(-delta * delay_ss)
        )
        discount_ll = (
            omega * np.exp(-beta * delay_ll)
            + (1.0 - omega) * np.exp(-delta * delay_ll)
        )
        value_ss = amount_ss * discount_ss
        value_ll = amount_ll * discount_ll
        return expit(m * (value_ll - value_ss))


class M6ConstantSensitivity(BaseChoiceModel):

    def __init__(self):
        super().__init__("m6")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        amount_ss, delay_ss, amount_ll, delay_ll = trials.T
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        m = parameters["m"]

        value_ss = amount_ss * np.exp(-(alpha * delay_ss) ** beta)
        value_ll = amount_ll * np.exp(-(alpha * delay_ll) ** beta)
        return expit(m * (value_ll - value_ss))


class M7AdditiveUtility(BaseChoiceModel):

    def __init__(self):
        super().__init__("m7")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        amount_ss, delay_ss, amount_ll, delay_ll = trials.T
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        delay_weight = parameters["lambda"]
        m = parameters["m"]

        value_ss = (
            amount_ss**alpha
            - delay_weight * delay_ss**beta
        )
        value_ll = (
            amount_ll**alpha
            - delay_weight * delay_ll**beta
        )
        return expit(m * (value_ll - value_ss))


class M8ProportionalDifference(BaseChoiceModel):

    def __init__(self):
        super().__init__("m8")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        amount_ss, delay_ss, amount_ll, delay_ll = trials.T
        delta = parameters["delta"]
        sigma = parameters["sigma"]

        amount_difference = (amount_ll - amount_ss) / amount_ll
        delay_difference = (delay_ll - delay_ss) / delay_ll
        d_ll = amount_difference - delay_difference
        return ndtr(sigma * (d_ll - delta))


def _positive_truncated_normal_cdf(x, mean, sd):
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    lower_mass = ndtr(-mean / sd)
    denominator = np.clip(1.0 - lower_mass, 1e-12, None)
    probability = (ndtr((x - mean) / sd) - lower_mass) / denominator
    return np.clip(probability, 0.0, 1.0)


def _normalized_heuristic_gaps(trials):
    amount_ss, delay_ss, amount_ll, delay_ll = trials.T
    x_amount = np.clip(
        (amount_ll - amount_ss) / HEURISTIC_AMOUNT_GAP_SCALE,
        0.0,
        1.0,
    )
    x_delay = np.clip(
        (delay_ll - delay_ss) / HEURISTIC_DELAY_GAP_SCALE,
        0.0,
        1.0,
    )
    return amount_ss, delay_ss, amount_ll, delay_ll, x_amount, x_delay


class M9PriorityLexicographicSemiorder(BaseChoiceModel):

    def __init__(self):
        super().__init__("m9")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        _, _, _, _, x_amount, x_delay = _normalized_heuristic_gaps(trials)
        g_amount = _positive_truncated_normal_cdf(
            x_amount,
            parameters["mu_amount"],
            parameters["sigma_threshold"],
        )
        g_delay = _positive_truncated_normal_cdf(
            x_delay,
            parameters["mu_delay"],
            parameters["sigma_threshold"],
        )
        p_amount_first = (
            g_amount + 0.5 * (1.0 - g_amount) * (1.0 - g_delay)
        )
        p_delay_first = (1.0 - g_delay) * (
            g_amount + 0.5 * (1.0 - g_amount)
        )
        q_amount_first = parameters["q_amount_first"]
        return (
            q_amount_first * p_amount_first
            + (1.0 - q_amount_first) * p_delay_first
        )


def _similarity_judgment_probabilities(trials, parameters):
    amount_ss, delay_ss, amount_ll, delay_ll, amount_difference, delay_difference = (
        _normalized_heuristic_gaps(trials)
    )
    amount_ratio = amount_ss / amount_ll
    delay_ratio = delay_ss / delay_ll
    tau = parameters["tau"]

    p_amount_similar_by_difference = expit(
        (parameters["theta_amount_diff"] - amount_difference) / tau
    )
    p_amount_similar_by_ratio = expit(
        (amount_ratio - parameters["theta_amount_ratio"]) / tau
    )
    s_amount = (
        p_amount_similar_by_difference
        + (1.0 - p_amount_similar_by_difference) * p_amount_similar_by_ratio
    )

    p_delay_similar_by_difference = expit(
        (parameters["theta_delay_diff"] - delay_difference) / tau
    )
    p_delay_similar_by_ratio = expit(
        (delay_ratio - parameters["theta_delay_ratio"]) / tau
    )
    s_delay = (
        p_delay_similar_by_difference
        + (1.0 - p_delay_similar_by_difference) * p_delay_similar_by_ratio
    )
    return s_amount, s_delay


class M10SimilarityHeuristic(BaseChoiceModel):

    def __init__(self):
        super().__init__("m10")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        s_amount, s_delay = _similarity_judgment_probabilities(
            trials,
            parameters,
        )

        return (
            (1.0 - s_amount) * s_delay
            + 0.5
            * (
                s_amount * s_delay
                + (1.0 - s_amount) * (1.0 - s_delay)
            )
        )


class M11ConstantLLProbability(BaseChoiceModel):

    def __init__(self):
        super().__init__("m11")

    def _calculate_p_ll(
        self,
        trials: np.ndarray,
        parameters: dict,
    ):
        p = float(parameters["p"])
        return np.full(trials.shape[0], p, dtype=np.float64)
