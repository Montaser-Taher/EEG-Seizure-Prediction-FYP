from pathlib import Path
import sys
import json
import random
import zipfile
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score


# PATH SETUP

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# RANDOM SEED SETUP

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
FINAL_THRESHOLD = THRESHOLD

EPOCHS = 30

EARLY_STOPPING_PATIENCE = 1
REDUCE_LR_PATIENCE = 2
MIN_LR = 1e-5

CLASS_WEIGHT_PREICTAL = 1.2
CLASS_WEIGHT_INTERICTAL = 1.0

# this boolean to only run the testing if the model alredy trained 
RUN_TRAINING = False

train_cache_dir = PROJECT_ROOT / "data" / "cached_train_lstm_seq5_batches_with_freq"
val_cache_dir = PROJECT_ROOT / "data" / "cached_val_lstm_seq5_batches_with_freq"
test_cache_dir = Path("/mnt/d/EEG_FYP_DATA/data/cached_test_lstm_seq5_batches_with_freq_fast")

models_dir = PROJECT_ROOT / "models" / "cnn_lstm_frequency"
results_dir = PROJECT_ROOT / "results" / "cnn_lstm_frequency"

models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)


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

            try:
                data = np.load(batch_file)

                X = data["X"].astype(np.float32)
                freq = data["freq"].astype(np.float32)
                y = data["y"].astype(np.float32)

            except zipfile.BadZipFile:
                print(f"SKIPPING CORRUPTED FILE: {batch_file}")
                continue

            except Exception as e:
                print(f"SKIPPING BAD FILE: {batch_file}")
                print("Reason:", e)
                continue

            if X.ndim != 4:
                raise ValueError(f"Bad X shape in {batch_file.name}: {X.shape}")

            if freq.ndim != 3:
                raise ValueError(f"Bad freq shape in {batch_file.name}: {freq.shape}")

            if X.shape[1:] != (SEQUENCE_LENGTH, TIMEPOINTS, CHANNELS):
                raise ValueError(f"Unexpected X shape in {batch_file.name}: {X.shape}")

            if freq.shape[1:] != (SEQUENCE_LENGTH, FREQ_FEATURES):
                raise ValueError(f"Unexpected freq shape in {batch_file.name}: {freq.shape}")

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

    # CNN PER WINDOW BRANCH

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

    x = tf.keras.layers.TimeDistributed(
        cnn_window_model,
        name="time_distributed_cnn"
    )(eeg_input)

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

    # COMBINE BRANCHES

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


# EVALUATION FUNCTION

def evaluate_model_on_cache(
    model,
    cache_dir,
    batch_files,
    pattern,
    threshold,
    split_name,
    results_dir
):
    generator = cached_sequence_batch_generator(
        cache_dir,
        shuffle=False,
        use_sample_weights=False,
        pattern=pattern
    )

    y_true_all = []
    y_prob_all = []

    for step in range(len(batch_files)):
        (X_batch, freq_batch), y_batch = next(generator)

        y_prob = model.predict(
            (X_batch, freq_batch),
            verbose=0
        ).ravel()

        y_true_all.extend(y_batch.astype(int))
        y_prob_all.extend(y_prob)

        if step % 100 == 0:
            print(f"{split_name}: evaluated batch [{step + 1}/{len(batch_files)}]")

    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)

    y_pred_all = (y_prob_all >= threshold).astype(int)

    tp = np.sum((y_true_all == 1) & (y_pred_all == 1))
    tn = np.sum((y_true_all == 0) & (y_pred_all == 0))
    fp = np.sum((y_true_all == 0) & (y_pred_all == 1))
    fn = np.sum((y_true_all == 1) & (y_pred_all == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    pr_auc = average_precision_score(y_true_all, y_prob_all)

    print("=" * 100)
    print(f"{split_name.upper()} EVALUATION USING FIXED THRESHOLD")
    print("=" * 100)
    print(f"Threshold: {threshold:.4f}")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    results = {
        "fixed_threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc)
    }

    results_path = results_dir / f"cnn_lstm_frequency_{split_name.lower()}_results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved {split_name} results to:", results_path)

    # Confusion matrix heatmap
    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"CNN + LSTM + Frequency {split_name} Confusion Matrix")
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

    heatmap_path = results_dir / f"cnn_lstm_frequency_{split_name.lower()}_confusion_matrix.png"

    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"Saved {split_name} confusion matrix to:", heatmap_path)

    return results


# LOAD CACHE FILES

train_batch_files = sorted(train_cache_dir.glob("train_batch_*.npz"))
val_batch_files = sorted(val_cache_dir.glob("val_batch_*.npz"))
test_batch_files = sorted(test_cache_dir.glob("test_batch_*.npz"))

print("Training LSTM batches:", len(train_batch_files))
print("Validation LSTM batches:", len(val_batch_files))
print("Testing LSTM batches:", len(test_batch_files))

if len(train_batch_files) == 0:
    raise ValueError("No training LSTM sequence batches found.")

if len(val_batch_files) == 0:
    raise ValueError("No validation LSTM sequence batches found.")

if len(test_batch_files) == 0:
    raise ValueError("No testing LSTM sequence batches found. Check the fast test cache folder.")


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
        "val_cache_dir": str(val_cache_dir),
        "test_cache_dir": str(test_cache_dir)
    }
}

with open(results_dir / "cnn_lstm_frequency_hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f, indent=4)

with open(results_dir / "cnn_lstm_frequency_model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

with open(results_dir / "cnn_lstm_frequency_model_architecture.json", "w") as f:
    f.write(model.to_json())


# TRAINING

best_model_path = models_dir / "cnn_lstm_frequency_best.keras"
final_model_path = models_dir / "cnn_lstm_frequency_final.keras"
log_path = results_dir / "cnn_lstm_frequency_training_log.csv"

if RUN_TRAINING:

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

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
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

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_batch_files),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_batch_files),
        callbacks=callbacks
    )

    model.save(final_model_path)

    print("Saved final CNN + LSTM + Frequency model:", final_model_path)

else:
    print("RUN_TRAINING is False, skipping training.")


# LOAD BEST MODEL

if not best_model_path.exists():
    raise ValueError(f"Best model not found: {best_model_path}")

best_model = tf.keras.models.load_model(best_model_path)


# VALIDATION EVALUATION

validation_results = evaluate_model_on_cache(
    model=best_model,
    cache_dir=val_cache_dir,
    batch_files=val_batch_files,
    pattern="val_batch_*.npz",
    threshold=FINAL_THRESHOLD,
    split_name="Validation",
    results_dir=results_dir
)


# TEST EVALUATION

test_results = evaluate_model_on_cache(
    model=best_model,
    cache_dir=test_cache_dir,
    batch_files=test_batch_files,
    pattern="test_batch_*.npz",
    threshold=FINAL_THRESHOLD,
    split_name="Test",
    results_dir=results_dir
)


# TRAINING LOG SUMMARY + PLOTS

if log_path.exists():

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

    auc_curve_path = results_dir / "cnn_lstm_frequency_pr_auc_curve.png"
    plt.savefig(auc_curve_path, dpi=300)
    plt.close()

    print("Saved PR-AUC curve to:", auc_curve_path)

else:
    print("Training log not found, skipping training plots.")

print("=" * 80)
print("DONE")
print("=" * 80)