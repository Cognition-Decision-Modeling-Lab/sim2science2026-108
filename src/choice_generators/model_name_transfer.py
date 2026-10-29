MODEL_METADATA = {
    "m1": {
        "class": 1,
        "class_name": "Discounted-value",
        "new_name": "M1-1",
        "short_name": "HYP",
        "full_name": "Hyperbolic Discounting",
        "formula": r"V_j=\frac{A_j}{1+kD_j},\qquad P(LL)=\operatorname{logit}^{-1}\!\left[m(V_{LL}-V_{SS})\right]",
    },
    "m2": {
        "class": 1,
        "class_name": "Discounted-value",
        "new_name": "M1-2",
        "short_name": "GHYP",
        "full_name": "Generalized Hyperbolic Discounting",
        "formula": r"V_j=\frac{A_j}{1+(kD_j)^s},\qquad P(LL)=\operatorname{logit}^{-1}\!\left[m(V_{LL}-V_{SS})\right]",
    },
    "m3": {
        "class": 2,
        "class_name": "Compensatory attribute-wise",
        "new_name": "M2-1",
        "short_name": "DDIFF",
        "full_name": "Direct Difference",
        "formula": (
            r"\Delta_A=A_{LL}^{u}-A_{SS}^{u},\qquad "
            r"\Delta_D=D_{LL}^{v}-D_{SS}^{v};\quad "
            r"d=w\Delta_A-(1-w)\Delta_D;\quad "
            r"\sigma_d^2=w\Delta_A^2+(1-w)\Delta_D^2-d^2;\quad "
            r"P(LL)=\Phi\!\left(\frac{d}{\sigma_d}\right)"
        ),
    },
    "m4": {
        "class": 1,
        "class_name": "Discounted-value",
        "new_name": "M1-3",
        "short_name": "EXP",
        "full_name": "Exponential Discounting",
        "formula": r"V_j=A_j e^{-kD_j},\qquad P(LL)=\operatorname{logit}^{-1}\!\left[m(V_{LL}-V_{SS})\right]",
    },
    "m5": {
        "class": 1,
        "class_name": "Discounted-value",
        "new_name": "M1-4",
        "short_name": "DEXP",
        "full_name": "Double Exponential Discounting",
        "formula": (
            r"\beta=\delta+\beta_{\mathrm{gap}};\qquad "
            r"V_j=A_j\left[\omega e^{-\beta D_j}+(1-\omega)e^{-\delta D_j}\right];\quad "
            r"P(LL)=\operatorname{logit}^{-1}\!\left[m(V_{LL}-V_{SS})\right]"
        ),
    },
    "m6": {
        "class": 1,
        "class_name": "Discounted-value",
        "new_name": "M1-5",
        "short_name": "CS",
        "full_name": "Constant Sensitivity",
        "formula": (
            r"V_j=A_j\exp\!\left[-(\alpha D_j)^\beta\right],\qquad "
            r"P(LL)=\operatorname{logit}^{-1}\!\left[m(V_{LL}-V_{SS})\right]"
        ),
    },
    "m7": {
        "class": 3,
        "class_name": "Additively separable utility",
        "new_name": "M3-1",
        "short_name": "ADD",
        "full_name": "Additive Utility",
        "formula": (
            r"V_j=A_j^\alpha-\lambda D_j^\beta,\qquad "
            r"P(LL)=\operatorname{logit}^{-1}\!\left[m(V_{LL}-V_{SS})\right]"
        ),
    },
    "m8": {
        "class": 2,
        "class_name": "Compensatory attribute-wise",
        "new_name": "M2-2",
        "short_name": "PDIFF",
        "full_name": "Proportional Difference",
        "formula": (
            r"\Delta_A=\frac{A_{LL}-A_{SS}}{A_{LL}},\qquad "
            r"\Delta_D=\frac{D_{LL}-D_{SS}}{D_{LL}};\quad "
            r"P(LL)=\Phi\!\left(\sigma\left[(\Delta_A-\Delta_D)-\delta\right]\right)"
        ),
    },
    "m9": {
        "class": 4,
        "class_name": "Heuristic / decision-tree",
        "new_name": "M4-1",
        "short_name": "PRIO",
        "full_name": "Priority-based Lexicographic Semiorder",
        "formula": (
            r"x_A=(A_{LL}-A_{SS})/490,\quad "
            r"x_D=(D_{LL}-D_{SS})/1200;\quad "
            r"g_k=F_{TN^+}(x_k;\mu_k,\sigma);\quad "
            r"P_A=g_A+\frac12(1-g_A)(1-g_D),\quad "
            r"P_D=(1-g_D)[g_A+\frac12(1-g_A)];\quad "
            r"P(LL)=qP_A+(1-q)P_D"
        ),
    },
    "m10": {
        "class": 4,
        "class_name": "Heuristic / decision-tree",
        "new_name": "M4-2",
        "short_name": "SIM",
        "full_name": "Similarity Heuristic",
        "formula": (
            r"d_A=(A_{LL}-A_{SS})/490,\quad r_A=A_{SS}/A_{LL};\quad "
            r"d_D=(D_{LL}-D_{SS})/1200,\quad r_D=D_{SS}/D_{LL};\quad "
            r"s_k=h_{k,d}+(1-h_{k,d})h_{k,r};\quad "
            r"P(LL)=(1-s_A)s_D+\frac12[s_As_D+(1-s_A)(1-s_D)]"
        ),
    },
    "m11": {
        "class": 5,
        "class_name": "Constant response",
        "new_name": "M5-1",
        "short_name": "CONST",
        "full_name": "Constant LL Probability",
        "formula": r"P(LL)=p",
    },
}

ORIGINAL_TO_NEW_NAME = {
    model_id: meta["new_name"] for model_id, meta in MODEL_METADATA.items()
}

NEW_NAME_TO_ORIGINAL = {
    meta["new_name"]: model_id for model_id, meta in MODEL_METADATA.items()
}

MODELS_BY_CLASS = {
    class_id: [
        model_id
        for model_id, meta in MODEL_METADATA.items()
        if meta["class"] == class_id
    ]
    for class_id in range(1, 6)
}


PAPER_FAMILY_METADATA = {
    "OW": {
        "family_name": "Option-wise value",
        "models": ("m1", "m2", "m4", "m5", "m6", "m7"),
    },
    "AW": {
        "family_name": "Attribute-wise comparison",
        "models": ("m3", "m8"),
    },
    "H": {
        "family_name": "Heuristic",
        "models": ("m9", "m10"),
    },
}


PAPER_MODEL_LABELS = {
    "m1": "OW1",
    "m2": "OW0",
    "m3": "AW1",
    "m4": "OW2",
    "m5": "OW3",
    "m6": "OW4",
    "m7": "OW5",
    "m8": "AW2",
    "m9": "H1",
    "m10": "H2",
    "m11": "CONST",
}

PAPER_MODEL_FAMILY = {
    model_id: family_code
    for family_code, family_meta in PAPER_FAMILY_METADATA.items()
    for model_id in family_meta["models"]
}

ORIGINAL_TO_PAPER_NAME = dict(PAPER_MODEL_LABELS)

PAPER_NAME_TO_ORIGINAL = {
    paper_name: model_id
    for model_id, paper_name in ORIGINAL_TO_PAPER_NAME.items()
}

MODELS_BY_PAPER_FAMILY = {
    family_code: list(family_meta["models"])
    for family_code, family_meta in PAPER_FAMILY_METADATA.items()
}

PAPER_MODEL_METADATA = {
    model_id: {
        "paper_name": PAPER_MODEL_LABELS[model_id],
        "paper_family": PAPER_MODEL_FAMILY.get(model_id),
        "paper_family_name": (
            PAPER_FAMILY_METADATA[PAPER_MODEL_FAMILY[model_id]]["family_name"]
            if model_id in PAPER_MODEL_FAMILY
            else "Constant-response control"
        ),
        "short_name": MODEL_METADATA[model_id]["short_name"],
        "full_name": MODEL_METADATA[model_id]["full_name"],
        "legacy_class": MODEL_METADATA[model_id]["class"],
        "legacy_name": MODEL_METADATA[model_id]["new_name"],
    }
    for model_id in MODEL_METADATA
}


def paper_model_label(model_id: str) -> str:
    return PAPER_MODEL_LABELS.get(model_id, model_id)


def paper_model_family(model_id: str) -> str | None:
    return PAPER_MODEL_FAMILY.get(model_id)


def paper_model_family_name(model_id: str) -> str | None:
    family_code = paper_model_family(model_id)
    if family_code is None:
        return None
    return PAPER_FAMILY_METADATA[family_code]["family_name"]


def paper_data_label(data_name: str) -> str:
    if data_name == "r1":
        return "Real"
    return paper_model_label(data_name)


if __name__ == "__main__":
    for model_id, meta in MODEL_METADATA.items():
        print(
            f"{model_id:>3} -> {meta['new_name']:<4} | "
            f"class={meta['class']} | "
            f"{meta['short_name']:<7} | "
            f"{meta['full_name']}"
        )
