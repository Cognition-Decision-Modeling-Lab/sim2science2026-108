CONFIG_NAME = "cfg3"

REPRESENTATION_HPARAMS = {
    "nn1": {
        "beta": 0.1,
    },
    "nn1b": {
        "beta": 0.1,
        "consistency_weight": 0.2,
    },
    "classifier": {},
    "nn1c": {
        "beta": 0.2,
        "classification_weight": 1.0,
    },
    "nn1bc": {
        "beta": 0.1,
        "consistency_weight": 0.2,
        "classification_weight": 1.0,
    },
    "parameter_decoder": {},
}
