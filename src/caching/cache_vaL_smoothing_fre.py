from pathlib import Path
import sys
import time
import numpy as np

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import TARGET_SFREQ
from utils import (
    load_prediction_npz_file,
    compute_frequency_features
)

# This cache is NOT balanced and NOT shuffled.
# It keeps each validation file in its original time order for smoothing.
cache_dir = PROJECT_ROOT / "data" / "cached_val_smoothing_frequency_by_file"
cache_dir.mkdir(parents=True, exist_ok=True)

# Set True if you want to delete old cached smoothing files and rebuild
rebuild_from_zero = False


def main():

    # =========================
    # GET VALIDATION FILES MANUALLY
    # =========================

    val_files = []

    val_files.extend(
        sorted((PROJECT_ROOT / "data" / "processed" / "group1" / "chb03").glob("*_processed.npz"))
    )

    val_files.extend(
        sorted((PROJECT_ROOT / "data" / "processed" / "group2" / "chb20").glob("*_processed.npz"))
    )

    if len(val_files) == 0:
        raise ValueError("No validation processed files found for chb03/chb20.")

    print("=" * 80)
    print("CACHE VALIDATION FREQUENCY FILES FOR SMOOTHING")
    print("=" * 80)
    print("Validation files:", len(val_files))
    print("Cache directory:", cache_dir)
    print("Sampling frequency:", TARGET_SFREQ)
    print("=" * 80)

    if rebuild_from_zero:
        for old_file in cache_dir.glob("*_smoothing_freq.npz"):
            old_file.unlink()
        print("Old smoothing frequency cache deleted.")

    start_time = time.time()

    total_windows = 0
    total_preictal = 0
    total_interictal = 0

    for file_i, file_path in enumerate(val_files, start=1):
        file_path = Path(file_path)

        save_path = cache_dir / f"{file_path.stem}_smoothing_freq.npz"

        if save_path.exists():
            print(f"[{file_i}/{len(val_files)}] Already cached: {save_path.name}")
            continue

        print("=" * 80)
        print(f"[{file_i}/{len(val_files)}] Loading: {file_path.name}")

        X, y = load_prediction_npz_file(file_path)

        if X.ndim != 3:
            raise ValueError(f"Expected X with 3 dimensions, got {X.shape}")

        # Convert to model shape: (windows, 1280, 17)
        if X.shape[1] == 17 and X.shape[2] == 1280:
            X_model = np.transpose(X, (0, 2, 1))
        elif X.shape[1] == 1280 and X.shape[2] == 17:
            X_model = X
        else:
            raise ValueError(
                f"Unexpected X shape: {X.shape}. "
                "Expected (windows, 17, 1280) or (windows, 1280, 17)."
            )

        y = y.astype(np.int64)

        print("X_model shape:", X_model.shape)
        print("y shape:", y.shape)
        print("Preictal:", int(np.sum(y == 1)))
        print("Interictal:", int(np.sum(y == 0)))

        freq_features = []

        print("Computing frequency features in time order...")

        for win_i in range(X_model.shape[0]):
            window = X_model[win_i]  # shape: (1280, 17)

            freq = compute_frequency_features(
                window,
                sfreq=TARGET_SFREQ
            )

            freq_features.append(freq)

            if win_i % 500 == 0 or win_i == X_model.shape[0] - 1:
                print(
                    f"  [{win_i + 1}/{X_model.shape[0]}] "
                    f"computed frequency features"
                )

        freq_features = np.stack(freq_features).astype(np.float32)

        np.savez(
            save_path,
            X=X_model.astype(np.float32),
            freq=freq_features,
            y=y,
            source_file=str(file_path)
        )

        total_windows += len(y)
        total_preictal += int(np.sum(y == 1))
        total_interictal += int(np.sum(y == 0))

        print("Saved:", save_path)
        print("Saved X shape:", X_model.shape)
        print("Saved freq shape:", freq_features.shape)
        print("Saved y shape:", y.shape)

    print("=" * 80)
    print("FINAL CACHE SUMMARY")
    print("=" * 80)
    print("Total windows processed:", total_windows)
    print("Total preictal:", total_preictal)
    print("Total interictal:", total_interictal)

    if total_preictal > 0:
        print("Interictal / Preictal ratio:", total_interictal / total_preictal)

    print(f"Time: {(time.time() - start_time) / 60:.2f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()