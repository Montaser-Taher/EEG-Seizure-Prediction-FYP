from pathlib import Path
import sys
import json
import random
from sklearn.metrics import average_precision_score
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


# PATH SETUP

# This is the main folder of my FYP project.
# I use this path so the script can find the data, models, results, and src folder.
PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")

# This is where my config.py and utility functions are stored.
SRC_DIR = PROJECT_ROOT / "src"

# This allows Python to import files from the src folder.
# Without this, Python may not find config.py or utils.py.
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# RANDOM SEED SETUP

# I set a random seed to make the results more reproducible.
# This helps reduce random changes between different training runs.
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# HYPERPARAMETERS

# Learning rate controls how fast the model updates its weights.
LEARNING_RATE = 0.00005

# Since preictal windows are much fewer than interictal windows,
# I give the preictal class a higher weight during training.
CLASS_WEIGHT_PREICTAL =2.0
CLASS_WEIGHT_INTERICTAL = 1.0

# This is a fixed threshold.
# The model outputs probabilities between 0 and 1.
# If the probability is >= 0.70, the window is classified as preictal.
THRESHOLD = 0.90

# I use the same fixed threshold for validation and testing.
# No automatic threshold tuning is used in this version.
FINAL_THRESHOLD = THRESHOLD

# L2 regularisation helps reduce overfitting.
L2_VALUE = 0.0002

# Maximum number of training epochs.
EPOCHS = 30

# Early stopping stops training if validation PR-AUC does not improve.
EARLY_STOPPING_PATIENCE = 3

# ReduceLROnPlateau reduces the learning rate if validation PR-AUC stops improving.
REDUCE_LR_PATIENCE = 2

# Minimum learning rate allowed.
MIN_LR = 1e-5


# GENERATOR FOR CACHED X + FREQUENCY

def cached_batch_generator_with_freq(
    cache_dir,
    shuffle=True,
    use_sample_weights=False,
    pattern="*.npz"
):
    # This function loads cached batches from disk.
    # Each batch contains:
    # X    = EEG time-domain windows
    # freq = extracted frequency features
    # y    = labels, where 0 = interictal and 1 = preictal

    cache_dir = Path(cache_dir)
    batch_files = sorted(cache_dir.glob(pattern))

    if len(batch_files) == 0:
        raise ValueError(f"No cached frequency batches found in {cache_dir}")

    print("=" * 60)
    print("Cached frequency batches found:", len(batch_files))
    print("Cache directory:", cache_dir)
    print("Pattern:", pattern)
    print("=" * 60)

    # Infinite generator because Keras keeps asking for batches during training.
    while True:

        # Training data is shuffled to reduce order bias.
        # Validation and testing are not shuffled.
        if shuffle:
            np.random.shuffle(batch_files)

        for batch_file in batch_files:
            data = np.load(batch_file)

            X = data["X"].astype(np.float32)
            freq = data["freq"].astype(np.float32)
            y = data["y"].astype(np.float32)

            # Sample weights are only used during training.
            # This makes preictal samples more important in the loss function.
            if use_sample_weights:
                sample_weights = np.where(
                    y == 1,
                    CLASS_WEIGHT_PREICTAL,
                    CLASS_WEIGHT_INTERICTAL
                ).astype(np.float32)

                yield (X, freq), y, sample_weights

            else:
                yield (X, freq), y


# MODEL

def build_cnn_with_frequency():

    # EEG input shape:
    # 1280 time samples = 5 seconds at 256 Hz
    # 17 EEG channels
    eeg_input = tf.keras.layers.Input(
        shape=(1280, 17),
        name="eeg_input"
    )

    # Frequency input shape:
    # 323 frequency features extracted from the same EEG window.
    freq_input = tf.keras.layers.Input(
        shape=(323,),
        name="frequency_input"
    )

    # L2 regularisation is applied to reduce overfitting.
    reg = tf.keras.regularizers.l2(L2_VALUE)

    # ========================================================
    # CNN BRANCH
    # ========================================================

    # This branch learns patterns directly from the raw EEG waveform.
    # Conv1D is suitable because EEG is a time-series signal.

    x = tf.keras.layers.Conv1D(
        32,
        7,
        activation="relu",
        padding="same",
        kernel_regularizer=reg
    )(eeg_input)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv1D(
        64,
        5,
        activation="relu",
        padding="same",
        kernel_regularizer=reg
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv1D(
        128,
        3,
        activation="relu",
        padding="same",
        kernel_regularizer=reg
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    # GlobalAveragePooling converts the feature maps into one compact vector.
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=reg,
        name="cnn_feature_dense"
    )(x)

    # ========================================================
    # FREQUENCY BRANCH
    # ========================================================

    # This branch learns from frequency-domain features.
    # These features can represent EEG rhythms and band-power information.
    f = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=reg
    )(freq_input)

    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(0.35)(f)

    # ========================================================
    # COMBINE CNN + FREQUENCY
    # ========================================================

    # The CNN features and frequency features are joined together.
    # This allows the model to use both time-domain and frequency-domain information.
    combined = tf.keras.layers.Concatenate()([x, f])

    combined = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=reg,
        name="combined_feature_dense"
    )(combined)

    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.30)(combined)

    # Sigmoid output gives one probability:
    # close to 0 = interictal
    # close to 1 = preictal
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(combined)

    model = tf.keras.Model(
        inputs=(eeg_input, freq_input),
        outputs=output,
        name="cnn_frequency"
    )

    # PR-AUC is monitored because the dataset is imbalanced.
    # Accuracy alone can be misleading in seizure prediction.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )

    return model


# LOAD CACHED FREQUENCY DATA

# These folders contain the cached training and validation batches.
train_cache_dir = PROJECT_ROOT / "data" / "cached_train_batches_with_freq_v3"
val_cache_dir = PROJECT_ROOT / "data" / "cached_val_batches_with_freq_v3"

train_batch_files = sorted(train_cache_dir.glob("train_batch_*.npz"))
val_batch_files = sorted(val_cache_dir.glob("val_batch_*.npz"))

print("Training frequency batches:", len(train_batch_files))
print("Validation frequency batches:", len(val_batch_files))

if len(train_batch_files) == 0:
    raise ValueError("No training frequency batches found.")

if len(val_batch_files) == 0:
    raise ValueError("No validation frequency batches found.")


# GENERATORS

# Training generator uses shuffling and sample weights.
train_generator = cached_batch_generator_with_freq(
    train_cache_dir,
    shuffle=True,
    use_sample_weights=True,
    pattern="train_batch_*.npz"
)

# Validation generator does not use shuffling.
# This keeps validation evaluation stable and consistent.
val_generator = cached_batch_generator_with_freq(
    val_cache_dir,
    shuffle=False,
    use_sample_weights=False,
    pattern="val_batch_*.npz"
)


# BUILD MODEL

model = build_cnn_with_frequency()
model.summary()


# FOLDERS

# All CNN + Frequency model files are saved in their own folder.
models_dir = PROJECT_ROOT / "models" / "cnn_frequency"
results_dir = PROJECT_ROOT / "results" / "cnn_frequency"

models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)


# SAVE MODEL STRUCTURE + HYPERPARAMETERS

# I save the hyperparameters so the experiment can be explained and repeated later.
hyperparameters = {
    "model_name": "cnn_frequency",

    "training": {
        "learning_rate": LEARNING_RATE,
        "fixed_threshold": THRESHOLD,
        "class_weight_preictal": CLASS_WEIGHT_PREICTAL,
        "class_weight_interictal": CLASS_WEIGHT_INTERICTAL,
        "epochs": EPOCHS,
        "steps_per_epoch": len(train_batch_files),
        "validation_steps": len(val_batch_files),
        "monitor": "val_pr_auc",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "min_lr": MIN_LR
    },

    "regularisation": {
        "l2": L2_VALUE,
        "batch_normalisation": True
    },

    "cache_paths": {
        "train_cache_dir": str(train_cache_dir),
        "val_cache_dir": str(val_cache_dir)
    }
}

with open(results_dir / "cnn_frequency_hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f, indent=4)

with open(results_dir / "cnn_frequency_model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

with open(results_dir / "cnn_frequency_model_architecture.json", "w") as f:
    f.write(model.to_json())

print("Saved hyperparameters to:", results_dir / "cnn_frequency_hyperparameters.json")
print("Saved model summary to:", results_dir / "cnn_frequency_model_summary.txt")
print("Saved model architecture to:", results_dir / "cnn_frequency_model_architecture.json")


# CALLBACKS

log_path = results_dir / "cnn_frequency_training_log.csv"

callbacks = [
    # Saves only the best model based on validation PR-AUC.
    tf.keras.callbacks.ModelCheckpoint(
        filepath=models_dir / "cnn_frequency_best.keras",
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Saves training history into a CSV file.
    tf.keras.callbacks.CSVLogger(log_path),

    # Stops training if validation PR-AUC does not improve.
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduces learning rate if validation PR-AUC stops improving.
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    )
]


# TRAIN

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_batch_files),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_batch_files),
    callbacks=callbacks
)


# SAVE FINAL MODEL

final_model_path = models_dir / "cnn_frequency_final.keras"
model.save(final_model_path)

print("Saved final CNN + Frequency model:", final_model_path)


# TRAINING LOG SUMMARY

df = pd.read_csv(log_path)

print("=" * 80)
print("CNN + FREQUENCY TRAINING SUMMARY")
print("=" * 80)


# EVALUATE BEST MODEL ON VALIDATION DATA

# I evaluate the best saved model, not just the last epoch.
best_model_path = models_dir / "cnn_frequency_best.keras"
best_model = tf.keras.models.load_model(best_model_path)

print("=" * 100)
print("EVALUATING BEST MODEL ON VALIDATION DATA")
print("=" * 100)

eval_val_generator = cached_batch_generator_with_freq(
    val_cache_dir,
    shuffle=False,
    use_sample_weights=False,
    pattern="val_batch_*.npz"
)

y_true_all = []
y_prob_all = []

for step in range(len(val_batch_files)):
    (X_batch, freq_batch), y_batch = next(eval_val_generator)

    y_prob = best_model.predict(
        (X_batch, freq_batch),
        verbose=0
    ).ravel()

    y_true_all.extend(y_batch.astype(int))
    y_prob_all.extend(y_prob)

y_true_all = np.array(y_true_all)
y_prob_all = np.array(y_prob_all)


# FIXED THRESHOLD VALIDATION EVALUATION

# This version does not use automatic threshold tuning.
# The same fixed threshold is used for validation and testing.
y_pred_all = (y_prob_all >= FINAL_THRESHOLD).astype(int)

tp = np.sum((y_true_all == 1) & (y_pred_all == 1))
tn = np.sum((y_true_all == 0) & (y_pred_all == 0))
fp = np.sum((y_true_all == 0) & (y_pred_all == 1))
fn = np.sum((y_true_all == 1) & (y_pred_all == 0))

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("Validation Confusion Matrix:")
print(f"TN: {tn} | FP: {fp}")
print(f"FN: {fn} | TP: {tp}")

print()
print("Validation Evaluation Using Fixed Threshold:")
print(f"Threshold: {FINAL_THRESHOLD:.4f}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# VALIDATION CONFUSION MATRIX HEATMAP

cm = np.array([
    [tn, fp],
    [fn, tp]
])

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - CNN + Frequency Validation")
plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(tick_marks, ["Interictal", "Preictal"])
plt.yticks(tick_marks, ["Interictal", "Preictal"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

heatmap_path = results_dir / "cnn_frequency_validation_confusion_matrix_heatmap.png"

plt.savefig(heatmap_path, dpi=300)
plt.close()

print("Saved validation confusion matrix heatmap to:", heatmap_path)


# SAVE VALIDATION RESULTS

evaluation_results = {
    "fixed_threshold": float(FINAL_THRESHOLD),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

validation_results_path = results_dir / "cnn_frequency_best_model_validation_evaluation.json"

with open(validation_results_path, "w") as f:
    json.dump(evaluation_results, f, indent=4)

print("Saved validation evaluation results to:", validation_results_path)


# SHOW TRAINING SUMMARY

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 100)
print("FULL CNN + FREQUENCY TRAINING SUMMARY")
print("=" * 100)
print(df)

print("=" * 100)
print("BEST VALIDATION EPOCH")
print("=" * 100)

best_epoch = df.loc[df["val_pr_auc"].idxmax()]
print(best_epoch)


# PLOT TRAINING CURVES

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["recall"], label="Train Recall")
plt.plot(df["epoch"], df["val_recall"], label="Validation Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("CNN + Frequency: Train vs Validation Recall")
plt.legend()
plt.grid(True)

recall_curve_path = results_dir / "cnn_frequency_recall_curve.png"
plt.savefig(recall_curve_path, dpi=300)
plt.close()

print("Saved recall curve to:", recall_curve_path)


plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["pr_auc"], label="Train PR-AUC")
plt.plot(df["epoch"], df["val_pr_auc"], label="Validation PR-AUC")
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.title("CNN + Frequency: Train vs Validation PR-AUC")
plt.legend()
plt.grid(True)

auc_curve_path = results_dir / "cnn_frequency_pr_auc_curve.png"
plt.savefig(auc_curve_path, dpi=300)
plt.close()

print("Saved PR-AUC curve to:", auc_curve_path)


# TEST DATA EVALUATION

print("=" * 100)
print("FINAL TEST SET EVALUATION")
print("=" * 100)

# The test cache contains unseen patient test batches.
# This data should not be used for training or threshold tuning.
test_cache_dir = PROJECT_ROOT / "data" / "cached_test_batches_with_freq_v3"

test_batch_files = sorted(
    test_cache_dir.glob("test_batch_*.npz")
)

print("Testing frequency batches:", len(test_batch_files))

if len(test_batch_files) == 0:
    raise ValueError("No testing frequency batches found.")

test_generator = cached_batch_generator_with_freq(
    test_cache_dir,
    shuffle=False,
    use_sample_weights=False,
    pattern="test_batch_*.npz"
)

best_model = tf.keras.models.load_model(best_model_path)

y_true_test = []
y_prob_test = []

for step in range(len(test_batch_files)):

    (X_batch, freq_batch), y_batch = next(test_generator)

    y_prob = best_model.predict(
        (X_batch, freq_batch),
        verbose=0
    ).ravel()

    y_true_test.extend(y_batch.astype(int))
    y_prob_test.extend(y_prob)

y_true_test = np.array(y_true_test)
y_prob_test = np.array(y_prob_test)


# FIXED THRESHOLD TEST EVALUATION

# The test set uses the same fixed threshold.
# This avoids choosing a threshold based on the test set.
y_pred_test = (y_prob_test >= FINAL_THRESHOLD).astype(int)

tp_test = np.sum((y_true_test == 1) & (y_pred_test == 1))
tn_test = np.sum((y_true_test == 0) & (y_pred_test == 0))
fp_test = np.sum((y_true_test == 0) & (y_pred_test == 1))
fn_test = np.sum((y_true_test == 1) & (y_pred_test == 0))

accuracy_test = (
    (tp_test + tn_test) /
    (tp_test + tn_test + fp_test + fn_test)
)

precision_test = (
    tp_test /
    (tp_test + fp_test + 1e-8)
)

recall_test = (
    tp_test /
    (tp_test + fn_test + 1e-8)
)

f1_test = (
    2 * precision_test * recall_test /
    (precision_test + recall_test + 1e-8)
)
pr_auc_test = average_precision_score(y_true_test, y_prob_test)
print("Test Confusion Matrix:")
print(f"TN: {tn_test} | FP: {fp_test}")
print(f"FN: {fn_test} | TP: {tp_test}")

print()
print("Test Evaluation Using Fixed Threshold:")
print(f"Threshold: {FINAL_THRESHOLD:.4f}")
print(f"Test Accuracy : {accuracy_test:.4f}")
print(f"Test Precision: {precision_test:.4f}")
print(f"Test Recall   : {recall_test:.4f}")
print(f"Test F1-score : {f1_test:.4f}")
print(f"Test PR-AUC   : {pr_auc_test:.4f}")


# SAVE TEST RESULTS

test_results = {
    "fixed_threshold": float(FINAL_THRESHOLD),
    "tn": int(tn_test),
    "fp": int(fp_test),
    "fn": int(fn_test),
    "tp": int(tp_test),
    "accuracy": float(accuracy_test),
    "precision": float(precision_test),
    "recall": float(recall_test),
    "f1_score": float(f1_test),
    "pr_auc": float(pr_auc_test)
}

test_results_path = results_dir / "cnn_frequency_test_results.json"

with open(test_results_path, "w") as f:
    json.dump(test_results, f, indent=4)

print("Saved test results to:", test_results_path)


# TEST CONFUSION MATRIX HEATMAP

cm_test = np.array([
    [tn_test, fp_test],
    [fn_test, tp_test]
])

plt.figure(figsize=(6, 5))
plt.imshow(cm_test, interpolation="nearest")
plt.title("CNN + Frequency Test Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(tick_marks, ["Interictal", "Preictal"])
plt.yticks(tick_marks, ["Interictal", "Preictal"])

for i in range(cm_test.shape[0]):
    for j in range(cm_test.shape[1]):
        plt.text(j, i, cm_test[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

test_heatmap_path = results_dir / "cnn_frequency_test_confusion_matrix.png"

plt.savefig(test_heatmap_path, dpi=300)
plt.close()

print("Saved test heatmap to:", test_heatmap_path)

print("=" * 80)
print("DONE")
print("=" * 80)