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

# SETTINGS

test_cache_dir = PROJECT_ROOT / "data" / "cached_test_batches_with_freq_v3"
test_cache_dir.mkdir(parents=True, exist_ok=True)

progress_path = test_cache_dir / "test_cache_progress.json"

BATCH_SIZE = 32
REBUILD_FROM_ZERO = False


# HELPER FUNCTIONS

def load_progress():
    if progress_path.exists():
        with open(progress_path, "r") as f:
            return json.load(f)

    return {
        "completed_files": [],
        "next_batch_index": 0,
        "total_windows_saved": 0,
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
    # Save to temp file first, then rename.
    # This avoids broken/corrupted .npz files if the script crashes while saving.
    temp_path = save_path.with_suffix(".tmp.npz")

    np.savez(
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

    print("Old test cache and progress file deleted.")

# MAIN

def main():
    if REBUILD_FROM_ZERO:
        clear_cache()

    # Delete any temporary file left from a previous crash
    for tmp_file in test_cache_dir.glob("test_batch_*.tmp.npz"):
        tmp_file.unlink()

    progress = load_progress()

    train_files, val_files, test_files = get_final17_prediction_file_lists()

    completed_files = set(progress["completed_files"])
    batch_index = int(progress["next_batch_index"])

    print("=" * 80)
    print("CREATING / RESUMING TEST CACHED BATCHES WITH FREQUENCY")
    print("=" * 80)
    print("Test files:", len(test_files))
    print("Already completed files:", len(completed_files))
    print("Next batch index:", batch_index)
    print("Output folder:", test_cache_dir)
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
        print("Preictal:", int(np.sum(y == 1)))
        print("Interictal:", int(np.sum(y == 0)))

        file_windows_saved = 0
        file_preictal_saved = 0
        file_interictal_saved = 0

        for start in range(0, len(y), BATCH_SIZE):
            end = start + BATCH_SIZE

            X_batch = X[start:end]
            y_batch = y[start:end]

            if len(y_batch) == 0:
                continue

            save_path = test_cache_dir / f"test_batch_{batch_index:05d}.npz"

            if save_path.exists():
                print(f"Batch already exists, skipping: {save_path.name}")
                batch_index += 1
                continue

            print(f"Saving batch {batch_index:05d} | windows {start}:{end}")

            freq_batch = compute_freq_batch(X_batch)

            save_batch_atomic(
                save_path=save_path,
                X_batch=X_batch,
                freq_batch=freq_batch,
                y_batch=y_batch
            )

            preictal_count = int(np.sum(y_batch == 1))
            interictal_count = int(np.sum(y_batch == 0))

            file_windows_saved += len(y_batch)
            file_preictal_saved += preictal_count
            file_interictal_saved += interictal_count

            progress["total_windows_saved"] += int(len(y_batch))
            progress["total_preictal_saved"] += preictal_count
            progress["total_interictal_saved"] += interictal_count

            batch_index += 1
            progress["next_batch_index"] = batch_index

            # Save progress after every batch
            save_progress(progress)

            print(
                f"Saved {save_path.name} | "
                f"preictal={preictal_count} | "
                f"interictal={interictal_count}"
            )

        progress["completed_files"].append(file_key)
        progress["next_batch_index"] = batch_index

        # Save progress after every completed file
        save_progress(progress)

        print("-" * 80)
        print("Completed file:", file_path.name)
        print("File windows saved:", file_windows_saved)
        print("File preictal saved:", file_preictal_saved)
        print("File interictal saved:", file_interictal_saved)

    print("=" * 80)
    print("TEST CACHE DONE")
    print("=" * 80)
    print("Total windows saved:", progress["total_windows_saved"])
    print("Total preictal saved:", progress["total_preictal_saved"])
    print("Total interictal saved:", progress["total_interictal_saved"])
    print("Next batch index:", progress["next_batch_index"])
    print("Saved to:", test_cache_dir)


if __name__ == "__main__":
    main()