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


# A fixed random seed is used so that the results are more reproducible
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    # Deterministic TensorFlow operations are enabled when supported
    tf.config.experimental.enable_op_determinism()
    print("TensorFlow deterministic operations enabled.")
except Exception as e:
    print("Could not enable full TensorFlow determinism:", e)

print("Random seed set to:", SEED)


# The main project folder is defined here
PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

# The src folder is added so config.py and utils.py can be imported
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


# The main training hyperparameters are defined in one place
LEARNING_RATE = 0.0001
L2_VALUE = 0.0001
EPOCHS = 30
THRESHOLD = 0.20

DROPOUT_1 = 0.30
DROPOUT_2 = 0.35
DROPOUT_3 = 0.40
DROPOUT_DENSE = 0.45

# A higher weight is given to the preictal class because it is the warning class
CLASS_WEIGHT_INTERICTAL = 1.0
CLASS_WEIGHT_PREICTAL = 2.0


def build_baseline_cnn(input_shape):

    # A baseline CNN is built using only raw time-domain EEG windows
    model = tf.keras.Sequential([

        # The input shape is expected to be 1280 time samples and 17 channels
        tf.keras.layers.Input(shape=input_shape),

        # The first convolution block is used to learn simple local EEG patterns
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=7,
            activation="relu",
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(L2_VALUE)
        ),

        # Batch normalisation is used to make training more stable
        tf.keras.layers.BatchNormalization(),

        # Max pooling is used to reduce the feature size
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Dropout is used to reduce overfitting
        tf.keras.layers.Dropout(DROPOUT_1),

        # The second convolution block is used to learn deeper EEG patterns
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

        # The third convolution block is used to extract more complex features
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

        # Global average pooling is used instead of flattening to reduce parameters
        tf.keras.layers.GlobalAveragePooling1D(),

        # A small dense layer is used before the final prediction
        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_VALUE)
        ),

        tf.keras.layers.Dropout(DROPOUT_DENSE),

        # Sigmoid is used because the task is binary classification
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # The model is compiled with binary cross-entropy for interictal/preictal prediction
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",

            # Precision is measured at the selected threshold
            tf.keras.metrics.Precision(
                name="precision",
                thresholds=THRESHOLD
            ),

            # Recall is measured because missing preictal windows is important
            tf.keras.metrics.Recall(
                name="recall",
                thresholds=THRESHOLD
            ),

            # PR-AUC is used because the dataset is imbalanced
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            )
        ]
    )

    return model


def get_patient_id(file_path):

    # The patient ID is extracted from the file name, such as chb01 or chb05
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

    # All cached batch files are collected from the selected folder
    batch_files = sorted(cache_dir.glob(pattern))

    if len(batch_files) == 0:
        raise ValueError(f"No cached batches found in {cache_dir}")

    print("=" * 80)
    print("Cached training batches found:", len(batch_files))
    print("Cache directory:", cache_dir)
    print("Pattern:", pattern)
    print("Using sample weights")
    print("Interictal weight:", CLASS_WEIGHT_INTERICTAL)
    print("Preictal weight:", CLASS_WEIGHT_PREICTAL)
    print("=" * 80)

    while True:

        # The training batches are shuffled at the start of each pass
        if shuffle:
            np.random.shuffle(batch_files)

        for batch_file in batch_files:

            # Each cached batch is loaded from disk
            data = np.load(batch_file)

            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.float32)

            # More importance is given to preictal samples during training
            sample_weights = np.where(
                y == 1,
                CLASS_WEIGHT_PREICTAL,
                CLASS_WEIGHT_INTERICTAL
            ).astype(np.float32)

            yield X, y, sample_weights


# The cached training and validation folders are selected
train_cache_dir = PROJECT_ROOT / "data" / "cached_train_batches"
val_cache_dir = PROJECT_ROOT / "data" / "cached_val_batches"

# The cached batch files are loaded using their file patterns
train_batch_files = sorted(train_cache_dir.glob("train_batch_*.npz"))
val_batch_files = sorted(val_cache_dir.glob("val_batch_*.npz"))

print("Total cached training batches:", len(train_batch_files))
print("Total cached validation batches:", len(val_batch_files))

# Training cannot continue if cached training batches are missing
if len(train_batch_files) == 0:
    raise ValueError("No cached training batches found. Run cache_batches.py first.")

# Validation cannot continue if cached validation batches are missing
if len(val_batch_files) == 0:
    raise ValueError("No cached validation batches found. Run cache_val_batches.py first.")

# One batch is loaded from each split to check the saved shapes
sample_train = np.load(train_batch_files[0])
sample_val = np.load(val_batch_files[0])

print("Example train X shape:", sample_train["X"].shape)
print("Example train y shape:", sample_train["y"].shape)
print("Example val X shape:", sample_val["X"].shape)
print("Example val y shape:", sample_val["y"].shape)

# The total number of windows is estimated from the number of cached batches
print("Training windows:", len(train_batch_files) * sample_train["X"].shape[0])
print("Validation windows:", len(val_batch_files) * sample_val["X"].shape[0])


# The training generator is created with custom sample weights
train_generator = cached_batch_generator_with_custom_weights(
    train_cache_dir,
    shuffle=True,
    pattern="train_batch_*.npz"
)

# The validation generator is used without sample weights
val_generator = cached_batch_generator(
    val_cache_dir,
    shuffle=False,
    use_sample_weights=False,
    pattern="val_batch_*.npz"
)


# The baseline CNN is created using the final EEG input shape
model = build_baseline_cnn(input_shape=(1280, 17))

# The model structure is printed to check the architecture
model.summary()


# A separate folder is created for the baseline model results
results_dir = PROJECT_ROOT / "results"
baseline_results_dir = results_dir / "basemodel"
baseline_results_dir.mkdir(parents=True, exist_ok=True)


# The hyperparameters are saved so the experiment can be documented later
hyperparameters = {
    "model_name": "baseline_model",
    "model_type": "Conv1D CNN",
    "task": "binary seizure prediction",
    "class_0": "interictal",
    "class_1": "preictal",
    "uses_frequency_features": False,
    "input_domain": "time-domain EEG only",
    "input_shape": [1280, 17],
    "window_size_sec": WINDOW_SIZE_SEC,
    "target_sampling_frequency": TARGET_SFREQ,
    "channels": FINAL_17_CHANNELS,
    "learning_rate": LEARNING_RATE,
    "l2_regularization": L2_VALUE,
    "epochs": EPOCHS,
    "threshold": THRESHOLD,
    "conv_filters": [32, 64, 64],
    "kernel_sizes": [7, 5, 3],
    "dropout_values": [DROPOUT_1, DROPOUT_2, DROPOUT_3, DROPOUT_DENSE],
    "optimizer": "Adam",
    "loss": "binary_crossentropy",
    "monitor_metric": "val_pr_auc",
    "early_stopping_patience": 2,
    "reduce_lr_patience": 2,
    "class_weight_interictal": CLASS_WEIGHT_INTERICTAL,
    "class_weight_preictal": CLASS_WEIGHT_PREICTAL,
    "training_data": "training batches only",
    "validation_data": "validation batches for checkpointing and early stopping",
    "testing_data": "unseen test patients used once after training"
}

hyperparameter_path = baseline_results_dir / "baseline_model_hyperparameters.json"

with open(hyperparameter_path, "w") as f:
    json.dump(hyperparameters, f, indent=4)

print("Saved hyperparameters to:", hyperparameter_path)

# CALLBACKS

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

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]



# TRAIN MODEL


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_batch_files),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_batch_files),
    callbacks=callbacks
)



# SAVE FINAL MODEL


final_model_path = baseline_results_dir / "baseline_model_final.keras"
model.save(final_model_path)

print("Final baseline model saved to:", final_model_path)



# TRAINING LOG SUMMARY


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



# PLOT TRAINING AND VALIDATION CURVES


plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["recall"], label="Train Recall")
plt.plot(df["epoch"], df["val_recall"], label="Validation Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Train vs Validation Recall - Baseline Model")
plt.legend()
plt.grid(True)

recall_plot_path = baseline_results_dir / "baseline_model_recall_curve.png"
plt.savefig(recall_plot_path, dpi=300)
plt.close()

print("Saved recall curve to:", recall_plot_path)


plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["precision"], label="Train Precision")
plt.plot(df["epoch"], df["val_precision"], label="Validation Precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Train vs Validation Precision - Baseline Model")
plt.legend()
plt.grid(True)

precision_plot_path = baseline_results_dir / "baseline_model_precision_curve.png"
plt.savefig(precision_plot_path, dpi=300)
plt.close()

print("Saved precision curve to:", precision_plot_path)


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


plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss - Baseline Model")
plt.legend()
plt.grid(True)

loss_plot_path = baseline_results_dir / "baseline_model_loss_curve.png"
plt.savefig(loss_plot_path, dpi=300)
plt.close()

print("Saved loss curve to:", loss_plot_path)



# LOAD BEST VALIDATION MODEL


best_model_path = baseline_results_dir / "baseline_model_best_val_pr_auc.keras"

print("=" * 100)
print("LOADING BEST BASELINE MODEL FOR FINAL TESTING")
print("=" * 100)
print("Best model path:", best_model_path)

best_model = tf.keras.models.load_model(best_model_path)



# FINAL TEST EVALUATION ON UNSEEN TEST PATIENTS


print("=" * 100)
print("FINAL TEST EVALUATION ON UNSEEN TEST PATIENTS")
print("=" * 100)

try:
    train_files, val_files, test_files = get_final17_prediction_file_lists()
except OSError as e:
    print("=" * 80)
    print("TEST EVALUATION SKIPPED")
    print("Reason: processed data path is not accessible.")
    print("Error:", e)
    print("Training and validation outputs were already saved.")
    print("=" * 80)
    sys.exit(0)


test_y_true_all = []
test_y_prob_all = []

print("Running predictions on test patient files...")

for i, file_path in enumerate(test_files, start=1):
    X, y = load_prediction_npz_file(file_path)

    # Convert from (windows, channels, samples) to (windows, samples, channels)
    # because the CNN expects input shape (1280, 17).
    if X.ndim == 3 and X.shape[1] == 17:
        X = np.transpose(X, (0, 2, 1))

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    probs = best_model.predict(X, verbose=0).ravel()

    test_y_true_all.append(y)
    test_y_prob_all.append(probs)

    print(
        f"[{i}/{len(test_files)}] {Path(file_path).name} | "
        f"windows={len(y)} | "
        f"preictal={np.sum(y == 1)} | "
        f"interictal={np.sum(y == 0)}"
    )


test_y_true = np.concatenate(test_y_true_all)
test_y_prob = np.concatenate(test_y_prob_all)

test_y_pred = (test_y_prob >= THRESHOLD).astype(int)

test_accuracy = accuracy_score(test_y_true, test_y_pred)
test_precision = precision_score(test_y_true, test_y_pred, zero_division=0)
test_recall = recall_score(test_y_true, test_y_pred, zero_division=0)
test_f1 = f1_score(test_y_true, test_y_pred, zero_division=0)
test_pr_auc = average_precision_score(test_y_true, test_y_prob)

test_cm = confusion_matrix(test_y_true, test_y_pred, labels=[0, 1])
tn, fp, fn, tp = test_cm.ravel()

print("=" * 100)
print("FINAL TEST METRICS")
print("=" * 100)
print("Test samples:", len(test_y_true))
print("Test preictal:", int(np.sum(test_y_true == 1)))
print("Test interictal:", int(np.sum(test_y_true == 0)))
print("Threshold used:", THRESHOLD)
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-score:", test_f1)
print("PR-AUC:", test_pr_auc)
print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
print("Prediction probability range:", test_y_prob.min(), "to", test_y_prob.max())
print("Mean probability:", test_y_prob.mean())



# SAVE TEST METRICS


test_metrics = {
    "model_name": "baseline_model",
    "evaluation_split": "test",
    "threshold": THRESHOLD,
    "test_samples": int(len(test_y_true)),
    "test_preictal": int(np.sum(test_y_true == 1)),
    "test_interictal": int(np.sum(test_y_true == 0)),
    "accuracy": float(test_accuracy),
    "precision": float(test_precision),
    "recall": float(test_recall),
    "f1_score": float(test_f1),
    "pr_auc": float(test_pr_auc),
    "true_positive": int(tp),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_negative": int(tn),
    "min_probability": float(test_y_prob.min()),
    "max_probability": float(test_y_prob.max()),
    "mean_probability": float(test_y_prob.mean())
}

test_metrics_json_path = baseline_results_dir / "baseline_model_final_test_metrics.json"

with open(test_metrics_json_path, "w") as f:
    json.dump(test_metrics, f, indent=4)

print("Saved final test metrics JSON to:", test_metrics_json_path)



# SAVE TEST CONFUSION MATRIX


test_cm_df = pd.DataFrame(
    test_cm,
    index=["Actual Interictal", "Actual Preictal"],
    columns=["Predicted Interictal", "Predicted Preictal"]
)

test_cm_csv_path = baseline_results_dir / "baseline_model_test_confusion_matrix.csv"
test_cm_df.to_csv(test_cm_csv_path)

print("Saved test confusion matrix CSV to:", test_cm_csv_path)


plt.figure(figsize=(6, 5))
plt.imshow(test_cm, interpolation="nearest")
plt.title("Confusion Matrix - Baseline Model Test Set")
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Interictal", "Preictal"])
plt.yticks(tick_marks, ["Interictal", "Preictal"])

for i in range(test_cm.shape[0]):
    for j in range(test_cm.shape[1]):
        plt.text(
            j,
            i,
            test_cm[i, j],
            ha="center",
            va="center"
        )

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

test_cm_heatmap_path = baseline_results_dir / "baseline_model_test_confusion_matrix_heatmap.png"
plt.savefig(test_cm_heatmap_path, dpi=300)
plt.close()

print("Saved test confusion matrix heatmap to:", test_cm_heatmap_path)



# PATIENT-WISE TEST EVALUATION


patient_data = defaultdict(lambda: {"y_true": [], "y_prob": []})

for file_path, y, probs in zip(test_files, test_y_true_all, test_y_prob_all):
    patient_id = get_patient_id(file_path)

    patient_data[patient_id]["y_true"].append(y)
    patient_data[patient_id]["y_prob"].append(probs)

patient_results = []

for patient_id, data in patient_data.items():

    y_true_patient = np.concatenate(data["y_true"])
    y_prob_patient = np.concatenate(data["y_prob"])

    y_pred_patient = (y_prob_patient >= THRESHOLD).astype(int)

    patient_accuracy = accuracy_score(y_true_patient, y_pred_patient)
    patient_precision = precision_score(y_true_patient, y_pred_patient, zero_division=0)
    patient_recall = recall_score(y_true_patient, y_pred_patient, zero_division=0)
    patient_f1 = f1_score(y_true_patient, y_pred_patient, zero_division=0)

    if len(np.unique(y_true_patient)) > 1:
        patient_pr_auc = average_precision_score(y_true_patient, y_prob_patient)
    else:
        patient_pr_auc = None

    cm_patient = confusion_matrix(y_true_patient, y_pred_patient, labels=[0, 1])
    tn, fp, fn, tp = cm_patient.ravel()

    patient_results.append({
        "patient": patient_id,
        "windows": int(len(y_true_patient)),
        "preictal": int(np.sum(y_true_patient == 1)),
        "interictal": int(np.sum(y_true_patient == 0)),
        "threshold": THRESHOLD,
        "accuracy": float(patient_accuracy),
        "precision": float(patient_precision),
        "recall": float(patient_recall),
        "f1_score": float(patient_f1),
        "pr_auc": None if patient_pr_auc is None else float(patient_pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    })

patient_df = pd.DataFrame(patient_results)

patient_test_eval_path = baseline_results_dir / "baseline_model_patient_wise_test_results.csv"
patient_df.to_csv(patient_test_eval_path, index=False)

print("=" * 100)
print("PATIENT-WISE TEST RESULTS")
print("=" * 100)
print(patient_df)

print("Saved patient-wise test results to:", patient_test_eval_path)

print("=" * 80)
print("BASELINE MODEL TRAINING, VALIDATION, AND FINAL TESTING DONE")
print("=" * 80)