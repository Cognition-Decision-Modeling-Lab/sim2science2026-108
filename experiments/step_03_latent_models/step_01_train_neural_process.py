from dataclasses import replace
from functools import partial

import torch

from common import (
    LATENT_DIM,
    SEED,
    load_preprocessed_experiment,
    output_root,
)
from neural_processes_choice import (
    ChoiceNPConfig,
    ChoiceNPTrainingConfig,
    save_training_result,
    train_choice_neural_process,
)


BETA_VALUES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0)
BETA_VALUES = (0.1, 0.2)
TRIAL_DATA_NAME = "general_400"


DATA_NAMES = ("m4", "m5", "m6", "m7", "m8")

CONTEXT_MIN_FRACTION = 0.7
CONTEXT_MAX_FRACTION = 0.99
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
LEARNING_RATE = 1e-4
MAX_EPOCHS = 1000
PATIENCE = 20
OVERWRITE_EXISTING = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY_EPOCHS = 10

MODEL_CONFIG = ChoiceNPConfig(latent_dim=LATENT_DIM)
BASE_TRAINING_CONFIG = ChoiceNPTrainingConfig(
    beta=0.0,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    context_min_fraction=CONTEXT_MIN_FRACTION,
    context_max_fraction=CONTEXT_MAX_FRACTION,
    patience=PATIENCE,
    normalize_trials=False,
    seed=SEED,
)


def report_epoch(beta, epoch, train_metrics, validation_metrics):
    if epoch == 0 or (epoch + 1) % LOG_EVERY_EPOCHS == 0:
        print(
            f"b={beta:g} e={epoch + 1} "
            f"tloss={train_metrics['loss']:.5f} "
            f"trec={train_metrics['reconstruction']:.5f} "
            f"tkl={train_metrics['kl']:.5f} "
            f"tacc={train_metrics['accuracy']:.4f} "
            f"vloss={validation_metrics['loss']:.5f} "
            f"vrec={validation_metrics['reconstruction']:.5f} "
            f"vkl={validation_metrics['kl']:.5f} "
            f"vacc={validation_metrics['accuracy']:.4f}"
        )


experiment = load_preprocessed_experiment(
    TRIAL_DATA_NAME,
    DATA_NAMES,
    SEED,
)
print(f"Device: {DEVICE}")
for beta in BETA_VALUES:
    checkpoint_path = (
        output_root(
            "neural_process",
            TRIAL_DATA_NAME,
            DATA_NAMES,
        )
        / f"b{beta:g}"
        / "checkpoint.pt"
    )
    if checkpoint_path.exists() and not OVERWRITE_EXISTING:
        print(f"Skipping beta={beta:g}: {checkpoint_path}")
        continue
    training_config = replace(BASE_TRAINING_CONFIG, beta=beta)
    result = train_choice_neural_process(
        data=experiment.choice_data,
        model_config=MODEL_CONFIG,
        training_config=training_config,
        device=DEVICE,
        subject_splits=experiment.subject_splits,
        trial_standardizer=experiment.trial_standardizer,
        epoch_callback=partial(report_epoch, beta),
    )
    save_training_result(
        result,
        checkpoint_path,
        MODEL_CONFIG,
        training_config,
        extra_state=experiment.metadata,
    )
    print(
        f"beta={beta:g} test={result.test_metrics} "
        f"saved={checkpoint_path}"
    )
