from pathlib import Path
import sys
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# SETTINGS

SEQUENCE_LENGTH = 5
TIMEPOINTS = 1280
CHANNELS = 17
FREQ_FEATURES = 323

LEARNING_RATE = 0.00001
L2_VALUE = 0.003

THRESHOLD = 0.75
EPOCHS = 30

EARLY_STOPPING_PATIENCE = 1
REDUCE_LR_PATIENCE = 2
MIN_LR = 1e-5

CLASS_WEIGHT_PREICTAL = 1.2

CLASS_WEIGHT_INTERICTAL = 1.0

train_cache_dir = PROJECT_ROOT / "data" / "cached_train_lstm_seq5_batches_with_freq"
val_cache_dir = PROJECT_ROOT / "data" / "cached_val_lstm_seq5_batches_with_freq"

models_dir = PROJECT_ROOT / "models"
results_dir = PROJECT_ROOT / "results"

models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)


# GENERATOR

def cached_sequence_batch_generator(
    cache_dir,
    shuffle=True,
    use_sample_weights=False,
    pattern="*.npz"
):
    cache_dir = Path(cache_dir)
    batch_files = sorted(cache_dir.glob(pattern))

    if len(batch_files) == 0:
        raise ValueError(f"No cached sequence batches found in {cache_dir}")

    print("=" * 80)
    print("Cached sequence batches found:", len(batch_files))
    print("Cache directory:", cache_dir)
    print("Pattern:", pattern)
    print("=" * 80)

    while True:
        if shuffle:
            np.random.shuffle(batch_files)

        for batch_file in batch_files:
            data = np.load(batch_file)

            X = data["X"].astype(np.float32)
            freq = data["freq"].astype(np.float32)
            y = data["y"].astype(np.float32)

            # Expected:
            # X    = (batch, 5, 1280, 17)
            # freq = (batch, 5, 323)
            # y    = (batch,)

            if X.ndim != 4:
                raise ValueError(f"Bad X shape in {batch_file.name}: {X.shape}")

            if freq.ndim != 3:
                raise ValueError(f"Bad freq shape in {batch_file.name}: {freq.shape}")

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

def build_cnn_lstm_frequency_model():
    reg = tf.keras.regularizers.l2(L2_VALUE)

    eeg_input = tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH, TIMEPOINTS, CHANNELS),
        name="eeg_sequence_input"
    )

    freq_input = tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH, FREQ_FEATURES),
        name="frequency_sequence_input"
    )

    # CNN PER WINDOW
   
    cnn_window_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(TIMEPOINTS, CHANNELS)),

            tf.keras.layers.Conv1D(
                32,
                7,
                activation="relu",
                padding="same",
                kernel_regularizer=reg
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv1D(
                64,
                5,
                activation="relu",
                padding="same",
                kernel_regularizer=reg
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv1D(
                64,
                3,
                activation="relu",
                padding="same",
                kernel_regularizer=reg
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.30),

            tf.keras.layers.GlobalAveragePooling1D(),

            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=reg
            )
        ],
        name="cnn_window_feature_extractor"
    )

    # Apply CNN to each of the 5 windows
    x = tf.keras.layers.TimeDistributed(
        cnn_window_model,
        name="time_distributed_cnn"
    )(eeg_input)

    # x shape becomes: (batch, 5, 64)

    # EEG TEMPORAL LSTM BRANCH
    
    x = tf.keras.layers.LSTM(
        32,
        return_sequences=False,
        dropout=0.35,
        recurrent_dropout=0.0,
        kernel_regularizer=reg,
        name="eeg_lstm"
    )(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=reg
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    # FREQUENCY TEMPORAL BRANCH
    
    f = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=reg
        ),
        name="freq_time_dense"
    )(freq_input)

    f = tf.keras.layers.LSTM(
        32,
        return_sequences=False,
        dropout=0.35,
        recurrent_dropout=0.0,
        kernel_regularizer=reg,
        name="freq_lstm"
    )(f)

    f = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=reg
    )(f)

    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(0.35)(f)

    # COMBINE EEG CNN-LSTM + FREQUENCY LSTM
   
    combined = tf.keras.layers.Concatenate(name="combined_features")([x, f])

    combined = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=reg
    )(combined)

    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.40)(combined)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="prediction_output"
    )(combined)

    model = tf.keras.Model(
        inputs=(eeg_input, freq_input),
        outputs=output,
        name="cnn_lstm_frequency"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC")
        ]
    )

    return model

# LOAD CACHE FILES

train_batch_files = sorted(train_cache_dir.glob("train_batch_*.npz"))
val_batch_files = sorted(val_cache_dir.glob("val_batch_*.npz"))

print("Training LSTM batches:", len(train_batch_files))
print("Validation LSTM batches:", len(val_batch_files))

if len(train_batch_files) == 0:
    raise ValueError("No training LSTM sequence batches found.")

if len(val_batch_files) == 0:
    raise ValueError("No validation LSTM sequence batches found.")

# GENERATORS

train_generator = cached_sequence_batch_generator(
    train_cache_dir,
    shuffle=True,
    use_sample_weights=True,
    pattern="train_batch_*.npz"
)

val_generator = cached_sequence_batch_generator(
    val_cache_dir,
    shuffle=False,
    use_sample_weights=False,
    pattern="val_batch_*.npz"
)

# BUILD MODEL

model = build_cnn_lstm_frequency_model()
model.summary()

# SAVE MODEL STRUCTURE + HYPERPARAMETERS

hyperparameters = {
    "model_name": "cnn_lstm_frequency",
    "input_shapes": {
        "X": [SEQUENCE_LENGTH, TIMEPOINTS, CHANNELS],
        "freq": [SEQUENCE_LENGTH, FREQ_FEATURES]
    },
    "training": {
        "learning_rate": LEARNING_RATE,
        "threshold": THRESHOLD,
        "epochs": EPOCHS,
        "steps_per_epoch": len(train_batch_files),
        "validation_steps": len(val_batch_files),
        "class_weight_preictal": CLASS_WEIGHT_PREICTAL,
        "class_weight_interictal": CLASS_WEIGHT_INTERICTAL,
        "monitor": "val_pr_auc",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "min_lr": MIN_LR
    },
    "sequence": {
        "sequence_length": SEQUENCE_LENGTH,
        "window_seconds": 5,
        "context_seconds": SEQUENCE_LENGTH * 5
    },
    "cache_paths": {
        "train_cache_dir": str(train_cache_dir),
        "val_cache_dir": str(val_cache_dir)
    }
}

with open(results_dir / "cnn_lstm_frequency_hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f, indent=4)

with open(results_dir / "cnn_lstm_frequency_model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

with open(results_dir / "cnn_lstm_frequency_model_architecture.json", "w") as f:
    f.write(model.to_json())

# CALLBACKS

log_path = results_dir / "cnn_lstm_frequency_training_log.csv"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=models_dir / "cnn_lstm_frequency_best.keras",
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

# Training the model

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_batch_files),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_batch_files),
    callbacks=callbacks
)

# SAVE FINAL MODEL

final_model_path = models_dir / "cnn_lstm_frequency_final.keras"
model.save(final_model_path)

print("Saved final CNN + LSTM + Frequency model:", final_model_path)

# EVALUATE BEST MODEL ON VALIDATION DATA

best_model_path = models_dir / "cnn_lstm_frequency_best.keras"
best_model = tf.keras.models.load_model(best_model_path)

print("=" * 100)
print("EVALUATING BEST CNN + LSTM MODEL ON VALIDATION DATA")
print("=" * 100)

eval_val_generator = cached_sequence_batch_generator(
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

# F1-BALANCED THRESHOLD TUNING

# TARGET PRECISION + RECALL THRESHOLD TUNING

# FIXED THRESHOLD
# No automatic threshold tuning is used.
# The same threshold from the settings section is used for evaluation.

FINAL_THRESHOLD = THRESHOLD

print("=" * 100)
print("FIXED THRESHOLD USED")
print("=" * 100)
print(f"Threshold used: {FINAL_THRESHOLD:.4f}")


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

print("\nValidation Evaluation Using Fixed Threshold:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("=" * 100)
print("FINAL VALIDATION EVALUATION USING FIXED THRESHOLD")
print(f"Best threshold : {FINAL_THRESHOLD:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-score       : {f1:.4f}")

evaluation_results = {
    "manual_threshold": float(THRESHOLD),
    "threshold_selection_method": "fixed_threshold",
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

with open(results_dir / "cnn_lstm_frequency_best_model_validation_evaluation.json", "w") as f:
    json.dump(evaluation_results, f, indent=4)

print("Saved evaluation results to:", results_dir / "cnn_lstm_frequency_best_model_validation_evaluation.json")



# TRAINING LOG SUMMARY + PLOTS


df = pd.read_csv(log_path)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 100)
print("FULL CNN + LSTM TRAINING SUMMARY")
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
plt.title("CNN + LSTM + Frequency: Train vs Validation Recall")
plt.legend()
plt.grid(True)

recall_curve_path = results_dir / "cnn_lstm_frequency_recall_curve.png"
plt.savefig(recall_curve_path, dpi=300)
plt.close()

print("Saved recall curve to:", recall_curve_path)


plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["pr_auc"], label="Train PR-AUC")
plt.plot(df["epoch"], df["val_pr_auc"], label="Validation PR-AUC")
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.title("CNN + LSTM + Frequency: Train vs Validation PR-AUC")
plt.legend()
plt.grid(True)

auc_curve_path = results_dir / "cnn_lstm_frequency_auc_curve.png"
plt.savefig(auc_curve_path, dpi=300)
plt.close()

print("Saved AUC curve to:", auc_curve_path)

print("=" * 80)
print("DONE")
print("=" * 80)