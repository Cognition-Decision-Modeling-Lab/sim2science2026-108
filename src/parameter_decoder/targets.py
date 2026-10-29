from dataclasses import dataclass

import torch
from torch import Tensor


PARAMETER_TRANSFORMS_BY_MODEL = {
    "m1": ("log", "log"),
    "m2": ("log", "identity", "log"),
    "m3": ("identity", "identity", "logit"),
    "m4": ("log", "log"),
    "m5": ("log", "log", "logit", "log"),
    "m6": ("log", "identity", "log"),
    "m7": ("logit", "logit", "log", "log"),
    "m8": ("identity", "log"),
    "m9": (
        "logit",
        "logit",
        "log",
        "logit",
    ),
    "m10": (
        "logit",
        "logit",
        "logit",
        "logit",
        "log",
    ),
    "m11": ("logit",),
}


def _apply_transform(values: Tensor, transform: str) -> Tensor:
    if transform == "log":
        return torch.log(values.clamp_min(1e-8))
    if transform == "logit":
        return torch.logit(values.clamp(1e-6, 1.0 - 1e-6))
    return values


@dataclass(frozen=True, slots=True)
class ParameterStandardizer:
    model_name: str
    parameter_names: tuple[str, ...]
    transforms: tuple[str, ...]
    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(
        cls,
        values: Tensor,
        model_name: str,
        parameter_names: list[str] | tuple[str, ...],
        train_indices: Tensor,
    ) -> "ParameterStandardizer":
        means = []
        scales = []
        names = tuple(parameter_names)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("Values and parameter names do not align.")
        if model_name not in PARAMETER_TRANSFORMS_BY_MODEL:
            raise ValueError(f"Unknown model: {model_name}")
        transforms = PARAMETER_TRANSFORMS_BY_MODEL[model_name]
        if len(transforms) != len(names):
            raise ValueError("Parameter transforms and names do not align.")
        for column, transform in enumerate(transforms):
            transformed = _apply_transform(
                values[train_indices, column],
                transform,
            )
            means.append(transformed.mean())
            scales.append(
                transformed.var(unbiased=False).sqrt().clamp_min(1e-6)
            )
        return cls(
            model_name,
            names,
            transforms,
            torch.stack(means),
            torch.stack(scales),
        )

    def transform(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != len(self.parameter_names):
            raise ValueError("Values do not match this standardizer.")
        standardized = torch.zeros_like(values)
        for column, transform in enumerate(self.transforms):
            transformed = _apply_transform(values[:, column], transform)
            standardized[:, column] = (
                transformed - self.mean[column]
            ) / self.scale[column]
        return standardized
