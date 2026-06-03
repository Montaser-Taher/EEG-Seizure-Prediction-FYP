from pathlib import Path
import sys
import re
import json
from collections import defaultdict

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    average_precision_score
)

# RANDOM SEED SETUP

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("TensorFlow deterministic operations enabled.")
except Exception as e:
    print("Could not enable full TensorFlow determinism:", e)

print("Random seed set to:", SEED)
# Make Python able to find config.py and utils.py
PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (
    FINAL_17_CHANNELS,
    WINDOW_SIZE_SEC,
    TARGET_SFREQ
)

from utils import (
    cached_batch_generator,
    get_final17_prediction_file_lists,
    load_prediction_npz_file
)


# Main hyperparameter values
# I put them here so they can also be saved later in JSON
LEARNING_RATE = 0.0001
L2_VALUE = 0.0001
EPOCHS = 30
THRESHOLD = 0.20

DROPOUT_1 = 0.30
DROPOUT_2 = 0.35
DROPOUT_3 = 0.40
DROPOUT_DENSE = 0.45
CLASS_WEIGHT_INTERICTAL = 1.0
CLASS_WEIGHT_PREICTAL = 2.0


# This function builds the baseline CNN model for seizure prediction
# The model will learn to classify:
# 0 = interictal (normal brain activity)
# 1 = preictal (before seizure)

def build_baseline_cnn(input_shape):

    # Sequential means we stack layers one after another
    model = tf.keras.Sequential([

        # Input layer
        # input_shape = (timepoints, channels)
        # Example: (1280, 17)
        tf.keras.layers.Input(shape=input_shape),

        # This is the first layer that looks at the EEG signal
        # It tries to detect small/simple patterns like:
        # small spikes, short oscillations and tiny waveform changes
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=7,
            activation="relu",
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(L2_VALUE)
        ),

        # BatchNormalization makes training more stable
        # it normalizes the output so values are easier to learn from
        tf.keras.layers.BatchNormalization(),

        # MaxPooling reduces the size
        # keeps important info but removes unnecessary details
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Dropout randomly disables neurons during training
        # helps prevent overfitting
        tf.keras.layers.Dropout(DROPOUT_1),

        # This block combines features from Block 1
        # It starts learning stronger patterns like repeated spikes and abnormal rhythms
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=5,
            activation="relu",
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(L2_VALUE)
        ),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(DROPOUT_2),

        # This block learns higher-level patterns
        # These are more meaningful for prediction, such as preictal signatures
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(L2_VALUE)
        ),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(DROPOUT_3),

        # This layer converts all feature maps into one vector
        # It reduces the data size and keeps the most important information
        tf.keras.layers.GlobalAveragePooling1D(),

        # Dense layer combines all learned features
        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_VALUE)
        ),

        # More dropout to reduce overfitting before final prediction
        tf.keras.layers.Dropout(DROPOUT_DENSE),

        # Final output layer
        # 1 neuron because this is binary classification
        # sigmoid gives a value between 0 and 1
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),

            # PR-AUC is used because seizure prediction is imbalanced
            # It focuses more on precision and recall instead of overall accuracy
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )

    return model


def get_patient_id(file_path):
    match = re.search(r"chb\d+", Path(file_path).name)
    if match:
        return match.group(0)
    return "unknown"

def cached_batch_generator_with_custom_weights(
    cache_dir,
    shuffle=True,
    pattern="*.npz"
):
    cache_dir = Path(cache_dir)
    batch_files = sorted(cache_dir.glob(pattern))

    if len(batch_files) == 0:
        raise ValueError(f"No cached batches found in {cache_dir}")

    print("=" * 80)
    print("Cached batches found:", len(batch_files))
    print("Cache directory:", cache_dir)
    print("Pattern:", pattern)
    print("Using custom sample weights:")
    print("Interictal weight:", CLASS_WEIGHT_INTERICTAL)
    print("Preictal weight:", CLASS_WEIGHT_PREICTAL)
    print("=" * 80)

    while True:
        if shuffle:
            np.random.shuffle(batch_files)

        for batch_file in batch_files:
            data = np.load(batch_file)

            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.float32)

            sample_weights = np.where(
                y == 1,
                CLASS_WEIGHT_PREICTAL,
                CLASS_WEIGHT_INTERICTAL
            ).astype(np.float32)

            yield X, y, sample_weights

# cached folders
train_cache_dir = PROJECT_ROOT / "data" / "cached_train_batches"
val_cache_dir = PROJECT_ROOT / "data" / "cached_val_batches"

train_batch_files = sorted(train_cache_dir.glob("train_batch_*.npz"))
val_batch_files = sorted(val_cache_dir.glob("val_batch_*.npz"))

print("Total cached training batches:", len(train_batch_files))
print("Total cached validation batches:", len(val_batch_files))

if len(train_batch_files) == 0:
    raise ValueError("No cached training batches found. Run cache_batches.py first.")

if len(val_batch_files) == 0:
    raise ValueError("No cached validation batches found. Run cache_val_batches.py first.")

# Check one training batch
sample_train = np.load(train_batch_files[0])
print("Example train X shape:", sample_train["X"].shape)
print("Example train y shape:", sample_train["y"].shape)

# Check one validation batch
sample_val = np.load(val_batch_files[0])
print("Example val X shape:", sample_val["X"].shape)
print("Example val y shape:", sample_val["y"].shape)

print("Training windows:", len(train_batch_files) * sample_train["X"].shape[0])
print("Validation windows:", len(val_batch_files) * sample_val["X"].shape[0])

# generator from cached training batches
# Training batches are already balanced using the balance plan
# use_sample_weights=True gives more importance to preictal class 1
train_generator = cached_batch_generator_with_custom_weights(
    train_cache_dir,
    shuffle=True,
    pattern="train_batch_*.npz"
)

print("Using custom sample weights: interictal=1.0, preictal=2.0")

# generator from cached validation batches
# Validation batches are used for monitoring only
# Do not use sample weights for validation because validation should measure performance only
val_generator = cached_batch_generator(
    val_cache_dir,
    shuffle=False,
    use_sample_weights=False,
    pattern="val_batch_*.npz"
)

# build model
model = build_baseline_cnn(input_shape=(1280, 17))
model.summary()

# results folder
# Everything for this baseline model will be saved inside results/basemodel
results_dir = PROJECT_ROOT / "results"
baseline_results_dir = results_dir / "basemodel"

baseline_results_dir.mkdir(parents=True, exist_ok=True)

# Save hyperparameter values
# This is useful for the report because it records the exact model settings
hyperparameters = {
    "model_name": "baseline_model",
    "model_type": "Conv1D CNN",
    "task": "binary seizure prediction",
    "class_0": "interictal",
    "class_1": "preictal",
    "uses_frequency_features": False,
    "input_domain": "time domain EEG only",
    "input_shape": [1280, 17],
    "window_size_sec": WINDOW_SIZE_SEC,
    "target_sampling_frequency": TARGET_SFREQ,
    "channels": FINAL_17_CHANNELS,
    "learning_rate": LEARNING_RATE,
    "l2_regularization": L2_VALUE,
    "epochs": EPOCHS,
    "report_threshold": THRESHOLD,
    "conv_filters": [32, 64, 64],
    "kernel_sizes": [7, 5, 3],
    "dropout_values": [DROPOUT_1, DROPOUT_2, DROPOUT_3, DROPOUT_DENSE],
    "optimizer": "Adam",
    "loss": "binary_crossentropy",
    "monitor_metric": "val_pr_auc",
    "early_stopping_patience": 5,
    "reduce_lr_patience": 2,
    "class_weight_interictal": CLASS_WEIGHT_INTERICTAL,
    "class_weight_preictal": CLASS_WEIGHT_PREICTAL,
}

hyperparameter_path = baseline_results_dir / "baseline_model_hyperparameters.json"

with open(hyperparameter_path, "w") as f:
    json.dump(hyperparameters, f, indent=4)

print("Saved hyperparameters to:", hyperparameter_path)

# callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=baseline_results_dir / "baseline_model_best_val_pr_auc.keras",
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        filename=baseline_results_dir / "baseline_model_training_log.csv"
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),

    # If validation loss stops improving, learning rate is reduced
    # This helps the model fine tune instead of getting stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]

# train
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_batch_files),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_batch_files),
    callbacks=callbacks
)

# save final model
final_model_path = baseline_results_dir / "baseline_model_final.keras"
model.save(final_model_path)

print("Final baseline model saved to:", final_model_path)


# PLOT TRAINING CURVES

log_path = baseline_results_dir / "baseline_model_training_log.csv"

df = pd.read_csv(log_path)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 100)
print("FULL BASELINE CNN TRAINING SUMMARY")
print("=" * 100)
print(df)

print("=" * 100)
print("BEST VALIDATION EPOCH")
print("=" * 100)

best_epoch = df.loc[df["val_pr_auc"].idxmax()]
print(best_epoch)

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["recall"], label="Train Recall")
plt.plot(df["epoch"], df["val_recall"], label="Validation Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Train vs Validation Recall - Baseline Model")
plt.legend()
plt.grid(True)

plot_path = baseline_results_dir / "baseline_model_recall_curve.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print("Saved recall curve to:", plot_path)

# Plot PR-AUC curve during training
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["pr_auc"], label="Train PR-AUC")
plt.plot(df["epoch"], df["val_pr_auc"], label="Validation PR-AUC")
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.title("Train vs Validation PR-AUC - Baseline Model")
plt.legend()
plt.grid(True)

pr_auc_plot_path = baseline_results_dir / "baseline_model_pr_auc_curve.png"
plt.savefig(pr_auc_plot_path, dpi=300)
plt.close()

print("Saved PR-AUC curve to:", pr_auc_plot_path)


# FINAL VALIDATION EVALUATION ON CACHED VALIDATION BATCHES
# The model is evaluated using the fixed report threshold
# The best model is selected based on validation PR-AUC

best_model_path = baseline_results_dir / "baseline_model_best_val_pr_auc.keras"
print("LOADING BEST BASELINE MODEL FOR FINAL VALIDATION EVALUATION")
print("Best model path:", best_model_path)

best_model = tf.keras.models.load_model(best_model_path)

all_y_true = []
all_y_prob = []

print("Running predictions on cached validation batches...")

for i, batch_file in enumerate(val_batch_files):
    data = np.load(batch_file)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    probs = best_model.predict(X, verbose=0).ravel()

    all_y_true.append(y)
    all_y_prob.append(probs)

    if i % 50 == 0 or i == len(val_batch_files) - 1:
        print(f"[{i + 1}/{len(val_batch_files)}] predicted {batch_file.name}")

y_true = np.concatenate(all_y_true)
y_prob = np.concatenate(all_y_prob)

# Convert probabilities into final class predictions
# If probability is greater than or equal to THRESHOLD, predict preictal
y_pred = (y_prob >= THRESHOLD).astype(int)

validation_accuracy = accuracy_score(y_true, y_pred)
validation_precision = precision_score(y_true, y_pred, zero_division=0)
validation_recall = recall_score(y_true, y_pred, zero_division=0)
validation_f1 = f1_score(y_true, y_pred, zero_division=0)

# PR-AUC uses the true labels and probability scores
# It is better for imbalanced seizure prediction than ROC-AUC
validation_pr_auc = average_precision_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("FINAL VALIDATION METRICS")
print("Validation samples:", len(y_true))
print("Validation preictal:", int(np.sum(y_true == 1)))
print("Validation interictal:", int(np.sum(y_true == 0)))
print("Threshold used:", THRESHOLD)
print("Accuracy:", validation_accuracy)
print("Precision:", validation_precision)
print("Recall:", validation_recall)
print("F1-score:", validation_f1)
print("PR-AUC:", validation_pr_auc)
print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
print("Prediction probability range:", y_prob.min(), "to", y_prob.max())
print("Mean probability:", y_prob.mean())

# Save final validation metrics as JSON
# This makes it easy to show the final result settings and scores in the report
final_metrics = {
    "model_name": "baseline_model",
    "threshold": THRESHOLD,
    "validation_samples": int(len(y_true)),
    "validation_preictal": int(np.sum(y_true == 1)),
    "validation_interictal": int(np.sum(y_true == 0)),
    "accuracy": float(validation_accuracy),
    "precision": float(validation_precision),
    "recall": float(validation_recall),
    "f1_score": float(validation_f1),
    "pr_auc": float(validation_pr_auc),
    "true_positive": int(tp),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_negative": int(tn),
    "min_probability": float(y_prob.min()),
    "max_probability": float(y_prob.max()),
    "mean_probability": float(y_prob.mean())
}

final_metrics_json_path = baseline_results_dir / "baseline_model_final_validation_metrics.json"

with open(final_metrics_json_path, "w") as f:
    json.dump(final_metrics, f, indent=4)

print("Saved final validation metrics JSON to:", final_metrics_json_path)

# Save confusion matrix as CSV
# Rows are the real labels, columns are the predicted labels
cm_df = pd.DataFrame(
    cm,
    index=["Actual Interictal", "Actual Preictal"],
    columns=["Predicted Interictal", "Predicted Preictal"]
)

cm_csv_path = baseline_results_dir / "baseline_model_confusion_matrix.csv"
cm_df.to_csv(cm_csv_path)

print("Saved confusion matrix CSV to:", cm_csv_path)

# Save confusion matrix heatmap image
# This gives a clear visual figure for the report
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Baseline Model")
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Interictal", "Preictal"])
plt.yticks(tick_marks, ["Interictal", "Preictal"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

cm_heatmap_path = baseline_results_dir / "baseline_model_confusion_matrix_heatmap.png"
plt.savefig(cm_heatmap_path, dpi=300)
plt.close()

print("Saved confusion matrix heatmap to:", cm_heatmap_path)

# PATIENT-WISE VALIDATION EVALUATION
# This checks performance per validation patient
# It uses original validation files instead of only cached batches

print("PATIENT-WISE VALIDATION EVALUATION")

try:
    train_files, val_files, test_files = get_final17_prediction_file_lists()
except OSError as e:
    print("=" * 80)
    print("PATIENT-WISE VALIDATION SKIPPED")
    print("Reason: external processed data path is not accessible.")
    print("Error:", e)
    print("Training, validation metrics, JSON, CSV, and confusion matrix heatmap were already saved.")
    print("=" * 80)
    sys.exit(0)

patient_data = defaultdict(lambda: {"y_true": [], "y_prob": []})

print("Running patient-wise validation predictions from original validation files...")

for i, file_path in enumerate(val_files, start=1):
    X, y = load_prediction_npz_file(file_path)

    # Convert from (windows, channels, samples) to (windows, samples, channels)
    if X.ndim == 3 and X.shape[1] == 17:
        X = np.transpose(X, (0, 2, 1))

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    probs = best_model.predict(X, verbose=0).ravel()

    patient_id = get_patient_id(file_path)

    patient_data[patient_id]["y_true"].append(y)
    patient_data[patient_id]["y_prob"].append(probs)

    print(
        f"[{i}/{len(val_files)}] {Path(file_path).name} | "
        f"patient={patient_id} | windows={len(y)} | "
        f"preictal={np.sum(y == 1)} | interictal={np.sum(y == 0)}"
    )

patient_results = []

for patient_id, data in patient_data.items():

    y_true_patient = np.concatenate(data["y_true"])
    y_prob_patient = np.concatenate(data["y_prob"])

    y_pred_patient = (y_prob_patient >= THRESHOLD).astype(int)

    patient_accuracy = accuracy_score(y_true_patient, y_pred_patient)
    patient_precision = precision_score(y_true_patient, y_pred_patient, zero_division=0)
    patient_recall = recall_score(y_true_patient, y_pred_patient, zero_division=0)
    patient_f1 = f1_score(y_true_patient, y_pred_patient, zero_division=0)

    # PR-AUC is only valid when the patient has both classes
    if len(np.unique(y_true_patient)) > 1:
        patient_pr_auc = average_precision_score(y_true_patient, y_prob_patient)
    else:
        patient_pr_auc = None

    cm_patient = confusion_matrix(y_true_patient, y_pred_patient, labels=[0, 1])
    tn, fp, fn, tp = cm_patient.ravel()

    patient_results.append({
        "patient": patient_id,
        "windows": len(y_true_patient),
        "preictal": int(np.sum(y_true_patient == 1)),
        "interictal": int(np.sum(y_true_patient == 0)),
        "threshold": THRESHOLD,

        "accuracy": patient_accuracy,
        "precision": patient_precision,
        "recall": patient_recall,
        "f1_score": patient_f1,
        "pr_auc": patient_pr_auc,

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    })

patient_df = pd.DataFrame(patient_results)

patient_eval_path = baseline_results_dir / "baseline_model_patient_wise_validation_results.csv"
patient_df.to_csv(patient_eval_path, index=False)

print("PATIENT-WISE RESULTS")
print(patient_df)
print("Saved patient-wise validation results to:", patient_eval_path)
print("DONE")
