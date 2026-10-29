from dataclasses import replace
from functools import partial

import torch

from common import (
    LATENT_DIM,
    SEED,
    load_preprocessed_experiment,
    output_root,
    select_model_labeled_data,
)
from neural_processes_choice import (
    ChoiceNPTrainingConfig,
    save_training_result,
)
from neural_processes_choice_classifier import (
    ChoiceNPClassifierConfig,
    train_choice_neural_process_classifier,
)


BETA_VALUES = (0.2, 0.5, 1)
CLASSIFICATION_WEIGHT_VALUES = (0.2, 0.5, 1)
TRIAL_DATA_NAME = "general_400"
DATA_NAMES = ("m1", "m2", "m3")


CONTEXT_MIN_FRACTION = 0.7
CONTEXT_MAX_FRACTION = 0.99
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
MAX_EPOCHS = 1000
PATIENCE = 20
OVERWRITE_EXISTING = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY_EPOCHS = 10

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


def report_epoch(
    beta,
    classification_weight,
    epoch,
    train_metrics,
    validation_metrics,
):
    if epoch == 0 or (epoch + 1) % LOG_EVERY_EPOCHS == 0:
        print(
            f"b={beta:g} cls={classification_weight:g} e={epoch + 1} "
            f"tloss={train_metrics['loss']:.5f} "
            f"trec={train_metrics['reconstruction']:.5f} "
            f"tkl={train_metrics['kl']:.5f} "
            f"tcls={train_metrics['classification']:.5f} "
            f"tacc={train_metrics['accuracy']:.4f} "
            f"tmacc={train_metrics['model_accuracy']:.4f} "
            f"vloss={validation_metrics['loss']:.5f} "
            f"vrec={validation_metrics['reconstruction']:.5f} "
            f"vkl={validation_metrics['kl']:.5f} "
            f"vcls={validation_metrics['classification']:.5f} "
            f"vacc={validation_metrics['accuracy']:.4f} "
            f"vmacc={validation_metrics['model_accuracy']:.4f}"
        )


experiment = select_model_labeled_data(
    load_preprocessed_experiment(
        TRIAL_DATA_NAME,
        DATA_NAMES,
        SEED,
    )
)
MODEL_NAMES = experiment.data_names
MODEL_CONFIG = ChoiceNPClassifierConfig(
    latent_dim=LATENT_DIM,
    n_models=len(MODEL_NAMES),
)
print(f"Device: {DEVICE}")
print(f"Models: {MODEL_NAMES}")
for beta in BETA_VALUES:
    for classification_weight in CLASSIFICATION_WEIGHT_VALUES:
        checkpoint_path = (
            output_root(
                "nn1c",
                TRIAL_DATA_NAME,
                MODEL_NAMES,
            )
            / f"b{beta:g}_cls{classification_weight:g}"
            / "checkpoint.pt"
        )
        if checkpoint_path.exists() and not OVERWRITE_EXISTING:
            print(
                f"Skipping beta={beta:g} "
                f"classification={classification_weight:g}: "
                f"{checkpoint_path}"
            )
            continue
        training_config = replace(BASE_TRAINING_CONFIG, beta=beta)
        result = train_choice_neural_process_classifier(
            data=experiment.choice_data,
            labels=experiment.model_labels,
            model_config=MODEL_CONFIG,
            training_config=training_config,
            classification_weight=classification_weight,
            device=DEVICE,
            subject_splits=experiment.subject_splits,
            trial_standardizer=experiment.trial_standardizer,
            epoch_callback=partial(
                report_epoch, beta, classification_weight
            ),
        )
        save_training_result(
            result,
            checkpoint_path,
            MODEL_CONFIG,
            training_config,
            extra_state={
                **experiment.metadata,
                "classification_weight": classification_weight,
            },
        )
        print(
            f"beta={beta:g} classification={classification_weight:g} "
            f"test={result.test_metrics} saved={checkpoint_path}"
        )
