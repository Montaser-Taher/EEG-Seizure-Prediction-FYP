from pathlib import Path
import sys
import json
import numpy as np

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import TARGET_SFREQ
from utils import (
    get_final17_prediction_file_lists,
    load_prediction_npz_file,
    compute_frequency_features
)

# ============================================================
# SETTINGS
# ============================================================

test_cache_dir = PROJECT_ROOT / "data" / "cached_test_lstm_seq5_batches_with_freq"
test_cache_dir.mkdir(parents=True, exist_ok=True)

progress_path = test_cache_dir / "test_lstm_seq5_cache_progress.json"

SEQUENCE_LENGTH = 5
TIMEPOINTS = 1280
CHANNELS = 17
FREQ_FEATURES = 323

BATCH_SIZE = 32
REBUILD_FROM_ZERO = False

# Label strategy:
# The sequence label is taken from the LAST window in the sequence.
# This keeps the evaluation realistic because the model predicts based on past/current context.
LABEL_STRATEGY = "last_window"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_progress():
    if progress_path.exists():
        with open(progress_path, "r") as f:
            return json.load(f)

    return {
        "completed_files": [],
        "next_batch_index": 0,
        "total_sequences_saved": 0,
        "total_preictal_saved": 0,
        "total_interictal_saved": 0
    }


def save_progress(progress):
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=4)


def convert_to_model_shape(X):
    # Converts EEG windows to model shape: (windows, 1280, 17)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D X, got {X.shape}")

    if X.shape[1] == 17 and X.shape[2] == 1280:
        return np.transpose(X, (0, 2, 1))

    if X.shape[1] == 1280 and X.shape[2] == 17:
        return X

    raise ValueError(f"Unexpected X shape: {X.shape}")


def compute_freq_batch(X_batch):
    freq_list = []

    for i in range(X_batch.shape[0]):
        freq = compute_frequency_features(
            X_batch[i],
            sfreq=TARGET_SFREQ
        )

        freq_list.append(freq)

        if i % 10 == 0 or i == X_batch.shape[0] - 1:
            print(f"    frequency window [{i + 1}/{X_batch.shape[0]}]")

    return np.stack(freq_list).astype(np.float32)


def save_batch_atomic(save_path, X_batch, freq_batch, y_batch):
    temp_path = save_path.with_suffix(".tmp.npz")

    np.savez_compressed(
        temp_path,
        X=X_batch.astype(np.float32),
        freq=freq_batch.astype(np.float32),
        y=y_batch.astype(np.int64)
    )

    temp_path.replace(save_path)


def clear_cache():
    for old_file in test_cache_dir.glob("test_batch_*.npz"):
        old_file.unlink()

    for old_file in test_cache_dir.glob("test_batch_*.tmp.npz"):
        old_file.unlink()

    if progress_path.exists():
        progress_path.unlink()

    print("Old CNN-LSTM test cache and progress file deleted.")


# ============================================================
# MAIN
# ============================================================

def main():
    if REBUILD_FROM_ZERO:
        clear_cache()

    # Delete temporary files from previous crash
    for tmp_file in test_cache_dir.glob("test_batch_*.tmp.npz"):
        tmp_file.unlink()

    progress = load_progress()

    train_files, val_files, test_files = get_final17_prediction_file_lists()

    completed_files = set(progress["completed_files"])
    batch_index = int(progress["next_batch_index"])

    print("=" * 80)
    print("CREATING / RESUMING TEST CNN-LSTM SEQ5 CACHED BATCHES WITH FREQUENCY")
    print("=" * 80)
    print("Test files:", len(test_files))
    print("Already completed files:", len(completed_files))
    print("Next batch index:", batch_index)
    print("Output folder:", test_cache_dir)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Batch size:", BATCH_SIZE)
    print("Balanced:", False)
    print("Shuffled:", False)
    print("=" * 80)

    for file_i, file_path in enumerate(test_files, start=1):
        file_path = Path(file_path)
        file_key = str(file_path)

        if file_key in completed_files:
            print(f"[{file_i}/{len(test_files)}] Skipping already completed: {file_path.name}")
            continue

        print("=" * 80)
        print(f"[{file_i}/{len(test_files)}] Processing: {file_path.name}")

        X, y = load_prediction_npz_file(file_path)

        X = convert_to_model_shape(X).astype(np.float32)
        y = y.astype(np.int64)

        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Preictal windows:", int(np.sum(y == 1)))
        print("Interictal windows:", int(np.sum(y == 0)))

        if len(y) < SEQUENCE_LENGTH:
            print("Skipping file because it has fewer windows than sequence length.")
            progress["completed_files"].append(file_key)
            save_progress(progress)
            continue

        # Compute frequency once for the full file
        print("Computing frequency features for full file...")
        freq = compute_freq_batch(X)

        if freq.shape[1] != FREQ_FEATURES:
            raise ValueError(
                f"Frequency feature mismatch. Expected {FREQ_FEATURES}, got {freq.shape[1]}"
            )

        file_sequences_saved = 0
        file_preictal_saved = 0
        file_interictal_saved = 0

        batch_X = []
        batch_freq = []
        batch_y = []

        # Natural ordered sequences, stride = 1
        for start in range(0, len(y) - SEQUENCE_LENGTH + 1):
            end = start + SEQUENCE_LENGTH

            X_seq = X[start:end]
            freq_seq = freq[start:end]

            if LABEL_STRATEGY == "last_window":
                y_seq = y[end - 1]
            else:
                raise ValueError("Unsupported LABEL_STRATEGY")

            batch_X.append(X_seq)
            batch_freq.append(freq_seq)
            batch_y.append(y_seq)

            if len(batch_X) == BATCH_SIZE:
                save_path = test_cache_dir / f"test_batch_{batch_index:05d}.npz"

                if save_path.exists():
                    print(f"Batch already exists, skipping: {save_path.name}")
                    batch_index += 1
                    batch_X = []
                    batch_freq = []
                    batch_y = []
                    continue

                X_batch = np.array(batch_X, dtype=np.float32)
                freq_batch = np.array(batch_freq, dtype=np.float32)
                y_batch = np.array(batch_y, dtype=np.int64)

                save_batch_atomic(
                    save_path=save_path,
                    X_batch=X_batch,
                    freq_batch=freq_batch,
                    y_batch=y_batch
                )

                preictal_count = int(np.sum(y_batch == 1))
                interictal_count = int(np.sum(y_batch == 0))

                file_sequences_saved += len(y_batch)
                file_preictal_saved += preictal_count
                file_interictal_saved += interictal_count

                progress["total_sequences_saved"] += int(len(y_batch))
                progress["total_preictal_saved"] += preictal_count
                progress["total_interictal_saved"] += interictal_count

                batch_index += 1
                progress["next_batch_index"] = batch_index

                save_progress(progress)

                print(
                    f"Saved {save_path.name} | "
                    f"X={X_batch.shape} | "
                    f"freq={freq_batch.shape} | "
                    f"preictal={preictal_count} | "
                    f"interictal={interictal_count}"
                )

                batch_X = []
                batch_freq = []
                batch_y = []

        # Save remaining sequences from this file
        if len(batch_X) > 0:
            save_path = test_cache_dir / f"test_batch_{batch_index:05d}.npz"

            if not save_path.exists():
                X_batch = np.array(batch_X, dtype=np.float32)
                freq_batch = np.array(batch_freq, dtype=np.float32)
                y_batch = np.array(batch_y, dtype=np.int64)

                save_batch_atomic(
                    save_path=save_path,
                    X_batch=X_batch,
                    freq_batch=freq_batch,
                    y_batch=y_batch
                )

                preictal_count = int(np.sum(y_batch == 1))
                interictal_count = int(np.sum(y_batch == 0))

                file_sequences_saved += len(y_batch)
                file_preictal_saved += preictal_count
                file_interictal_saved += interictal_count

                progress["total_sequences_saved"] += int(len(y_batch))
                progress["total_preictal_saved"] += preictal_count
                progress["total_interictal_saved"] += interictal_count

                batch_index += 1
                progress["next_batch_index"] = batch_index

                save_progress(progress)

                print(
                    f"Saved {save_path.name} | "
                    f"X={X_batch.shape} | "
                    f"freq={freq_batch.shape} | "
                    f"preictal={preictal_count} | "
                    f"interictal={interictal_count}"
                )

        progress["completed_files"].append(file_key)
        progress["next_batch_index"] = batch_index
        save_progress(progress)

        print("-" * 80)
        print("Completed file:", file_path.name)
        print("File sequences saved:", file_sequences_saved)
        print("File preictal saved:", file_preictal_saved)
        print("File interictal saved:", file_interictal_saved)

    # Final summary file
    summary = {
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "timepoints": TIMEPOINTS,
        "channels": CHANNELS,
        "frequency_features": FREQ_FEATURES,
        "label_strategy": LABEL_STRATEGY,
        "balanced": False,
        "shuffle": False,
        "stride": 1,
        "total_sequences_saved": progress["total_sequences_saved"],
        "total_preictal_saved": progress["total_preictal_saved"],
        "total_interictal_saved": progress["total_interictal_saved"],
        "real_world_note": "This test cache is natural, unbalanced, and not shuffled. It is suitable for final real-world-style testing on unseen patients."
    }

    with open(test_cache_dir / "test_lstm_seq5_cache_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 80)
    print("CNN-LSTM TEST CACHE DONE")
    print("=" * 80)
    print("Total sequences saved:", progress["total_sequences_saved"])
    print("Total preictal saved:", progress["total_preictal_saved"])
    print("Total interictal saved:", progress["total_interictal_saved"])
    print("Next batch index:", progress["next_batch_index"])
    print("Saved to:", test_cache_dir)


if __name__ == "__main__":
    main()