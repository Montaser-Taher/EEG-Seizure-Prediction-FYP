from pathlib import Path
import sys
import json 
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from pathlib import Path
import sys
import json

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


LEARNING_RATE = 0.00005

CLASS_WEIGHT_PREICTAL = 2.0
CLASS_WEIGHT_INTERICTAL = 1.0

THRESHOLD = 0.70

L2_VALUE = 0.0002

EPOCHS = 30

EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
MIN_LR = 1e-5

# GENERATOR FOR CACHED X + FREQUENCY


def cached_batch_generator_with_freq(
    cache_dir,
    shuffle=True,
    use_sample_weights=False,
    pattern="*.npz"
):
    cache_dir = Path(cache_dir)
    batch_files = sorted(cache_dir.glob(pattern))

    if len(batch_files) == 0:
        raise ValueError(f"No cached frequency batches found in {cache_dir}")

    print("=" * 60)
    print("Cached frequency batches found:", len(batch_files))
    print("Cache directory:", cache_dir)
    print("Pattern:", pattern)
    print("=" * 60)

    while True:
        if shuffle:
            np.random.shuffle(batch_files)

        for batch_file in batch_files:
            data = np.load(batch_file)

            X = data["X"].astype(np.float32)
            freq = data["freq"].astype(np.float32)
            y = data["y"].astype(np.float32)

            if use_sample_weights:
                sample_weights = np.where(y == 1,CLASS_WEIGHT_PREICTAL,CLASS_WEIGHT_INTERICTAL).astype(np.float32)
                yield (X, freq), y, sample_weights
            else:
                yield (X, freq), y



# MODEL


def build_cnn_with_frequency():

    eeg_input = tf.keras.layers.Input(shape=(1280, 17), name="eeg_input")
    freq_input = tf.keras.layers.Input(shape=(323,), name="frequency_input")

    reg = tf.keras.regularizers.l2(L2_VALUE)


# CNN BRANCH


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

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=reg,
        name="cnn_feature_dense"
    )(x)


    # FREQUENCY BRANCH


    f = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=reg
    )(freq_input)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(0.35)(f)


    # COMBINE CNN + FREQUENCY


    combined = tf.keras.layers.Concatenate()([x, f])

    combined = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=reg,
        name="combined_feature_dense"
    )(combined)

    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.30)(combined)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(combined)

    model = tf.keras.Model(
        inputs=(eeg_input, freq_input),
        outputs=output,
        name="cnn_frequency"
    )

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


train_generator = cached_batch_generator_with_freq(
    train_cache_dir,
    shuffle=True,
    use_sample_weights=True,
    pattern="train_batch_*.npz"
)

val_generator = cached_batch_generator_with_freq(
    val_cache_dir,
    shuffle=False,
    use_sample_weights=False,
    pattern="val_batch_*.npz"
)



# MODEL


model = build_cnn_with_frequency()
model.summary()


# SAVE MODEL STRUCTURE + HYPERPARAMETERS


models_dir = PROJECT_ROOT / "models" / "cnn_frequency"
results_dir = PROJECT_ROOT / "results" / "cnn_frequency"

models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

hyperparameters = {
    "model_name": "cnn_frequency",

    "training": {
        "learning_rate": LEARNING_RATE,
        "precision_recall_threshold": THRESHOLD,
        "class_weight_preictal": CLASS_WEIGHT_PREICTAL,
        "class_weight_interictal": CLASS_WEIGHT_INTERICTAL,
        "epochs": EPOCHS,
        "steps_per_epoch": len(train_batch_files),
        "validation_steps": len(val_batch_files),
        "monitor": "val_pr_auc",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "min_lr": MIN_LR,
        "threshold": THRESHOLD
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


models_dir = PROJECT_ROOT / "models"
results_dir = PROJECT_ROOT / "results"

models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

log_path = results_dir / "cnn_frequency_training_log.csv"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=models_dir / "cnn_frequency_best.keras",
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(log_path),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),

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

    y_prob = best_model.predict((X_batch, freq_batch), verbose=0).ravel()

    y_true_all.extend(y_batch.astype(int))
    y_prob_all.extend(y_prob)

y_true_all = np.array(y_true_all)
y_prob_all = np.array(y_prob_all)


# AUTOMATIC THRESHOLD TUNING WITH MINIMUM RECALL


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_true_all, y_prob_all)

precisions = precisions[:-1]
recalls = recalls[:-1]

MIN_RECALL_REQUIRED = 0.65

valid_idxs = np.where(recalls >= MIN_RECALL_REQUIRED)[0]

if len(valid_idxs) > 0:
    best_idx = valid_idxs[np.argmax(precisions[valid_idxs])]
else:
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

FINAL_THRESHOLD = thresholds[best_idx]

y_pred_all = (y_prob_all >= FINAL_THRESHOLD).astype(int)

tp = np.sum((y_true_all == 1) & (y_pred_all == 1))
tn = np.sum((y_true_all == 0) & (y_pred_all == 0))
fp = np.sum((y_true_all == 0) & (y_pred_all == 1))
fn = np.sum((y_true_all == 1) & (y_pred_all == 0))

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("Confusion Matrix:")
print(f"TN: {tn} | FP: {fp}")
print(f"FN: {fn} | TP: {tp}")

print("\nValidation Evaluation Using Automatic Threshold:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("=" * 100)
print("AUTOMATIC THRESHOLD TUNING - RECALL-CONSTRAINED")
print("=" * 100)
print(f"Minimum recall required: {MIN_RECALL_REQUIRED:.2f}")
print(f"Best threshold         : {FINAL_THRESHOLD:.4f}")
print(f"Precision              : {precisions[best_idx]:.4f}")
print(f"Recall                 : {recalls[best_idx]:.4f}")


# CONFUSION MATRIX HEATMAP


# Create confusion matrix array
cm = np.array([
    [tn, fp],
    [fn, tp]
])

# Create folder path
cnn_frequency_results_dir = results_dir

# Create heatmap figure
plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix - CNN + Frequency")
plt.colorbar()

# Labels for rows and columns
tick_marks = np.arange(2)

plt.xticks(
    tick_marks,
    ["Interictal", "Preictal"]
)

plt.yticks(
    tick_marks,
    ["Interictal", "Preictal"]
)

# Put values inside heatmap boxes
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

# Save heatmap
heatmap_path = (
    cnn_frequency_results_dir /
    "cnn_frequency_confusion_matrix_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300
)

plt.close()

print("Saved confusion matrix heatmap to:", heatmap_path)

evaluation_results = {
    "manual_threshold": THRESHOLD,
    "automatic_best_threshold": float(FINAL_THRESHOLD),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open(results_dir / "cnn_frequency_best_model_validation_evaluation.json", "w") as f:
    json.dump(evaluation_results, f, indent=4)

print("Saved evaluation results to:", results_dir / "cnn_frequency_best_model_validation_evaluation.json")

#  SHOW ALL COLUMNS (important)
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
plt.ylabel("AUC")
plt.title("CNN + Frequency: Train vs Validation AUC")
plt.legend()
plt.grid(True)

auc_curve_path = results_dir / "cnn_frequency_auc_curve.png"
plt.savefig(auc_curve_path, dpi=300)
plt.close()

print("Saved AUC curve to:", auc_curve_path)

print("=" * 80)
print("DONE")
print("=" * 80)