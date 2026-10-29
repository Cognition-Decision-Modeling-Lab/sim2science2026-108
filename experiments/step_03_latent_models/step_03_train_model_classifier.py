import torch

from common import (
    LATENT_DIM,
    SEED,
    load_preprocessed_experiment,
    output_root,
    select_model_labeled_data,
)
from model_classifier import (
    ModelClassifierConfig,
    ModelClassifierTrainer,
    ModelClassifierTrainingConfig,
    save_classifier_training_result,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
MAX_EPOCHS = 1000
PATIENCE = 20
TRIAL_DATA_NAME = "general_400"


DATA_NAMES = ("m4", "m5", "m6", "m7", "m8")
TRAINING_CONFIG = ModelClassifierTrainingConfig(
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    normalize_trials=False,
    seed=SEED,
)


def report_epoch(epoch, train_metrics, validation_metrics):
    if epoch == 0 or (epoch + 1) % LOG_EVERY_EPOCHS == 0:
        print(
            f"e={epoch + 1} "
            f"tloss={train_metrics['loss']:.5f} "
            f"tacc={train_metrics['accuracy']:.4f} "
            f"vloss={validation_metrics['loss']:.5f} "
            f"vacc={validation_metrics['accuracy']:.4f}"
        )


experiment = select_model_labeled_data(
    load_preprocessed_experiment(
        TRIAL_DATA_NAME,
        DATA_NAMES,
        SEED,
    )
)
MODEL_NAMES = experiment.data_names
model_config = ModelClassifierConfig(
    latent_dim=LATENT_DIM,
    n_models=len(experiment.data_names),
)
print(f"Device: {DEVICE}")
trainer = ModelClassifierTrainer(
    model_config,
    TRAINING_CONFIG,
    DEVICE,
)
result = trainer.fit(
    data=experiment.choice_data,
    labels=experiment.model_labels,
    subject_splits=experiment.subject_splits,
    trial_standardizer=experiment.trial_standardizer,
    epoch_callback=report_epoch,
)
checkpoint_path = output_root(
    "model_classifier",
    TRIAL_DATA_NAME,
    MODEL_NAMES,
) / "checkpoint.pt"
save_classifier_training_result(
    result,
    checkpoint_path,
    model_config,
    TRAINING_CONFIG,
    extra_state=experiment.metadata,
)
print(f"Test metrics: {result.test_metrics}")
print(f"Saved to: {checkpoint_path}")
