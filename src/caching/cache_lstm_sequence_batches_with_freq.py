from pathlib import Path
import sys
import time
import numpy as np

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
EXTERNAL_ROOT = Path("/mnt/d/EEG_FYP_DATA")

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
# External drive root
EXTERNAL_ROOT = Path("/mnt/d/EEG_FYP_DATA")

# READ old window cache from project folder
window_train_cache_dir = PROJECT_ROOT / "data" / "cached_train_window_with_freq_by_file"
window_val_cache_dir = PROJECT_ROOT / "data" / "cached_val_window_with_freq_by_file"

# SAVE new seq10 cache to external drive
sequence_train_cache_dir = EXTERNAL_ROOT / "cached_train_lstm_seq10_stride10_batches_with_freq"
sequence_val_cache_dir = EXTERNAL_ROOT / "cached_val_lstm_seq10_stride10_batches_with_freq"

# Window-level cache already created before
# Keep reading these from your project unless you also moved them to D drive
window_train_cache_dir = PROJECT_ROOT / "data" / "cached_train_window_with_freq_by_file"
window_val_cache_dir = PROJECT_ROOT / "data" / "cached_val_window_with_freq_by_file"

# Save NEW sequence length 10 batches on external drive
sequence_train_cache_dir = EXTERNAL_ROOT / "cached_train_lstm_seq10_batches_with_freq"
sequence_val_cache_dir = EXTERNAL_ROOT / "cached_val_lstm_seq10_batches_with_freq"

window_train_cache_dir.mkdir(parents=True, exist_ok=True)
window_val_cache_dir.mkdir(parents=True, exist_ok=True)
sequence_train_cache_dir.mkdir(parents=True, exist_ok=True)
sequence_val_cache_dir.mkdir(parents=True, exist_ok=True)

# those setthing of the how is the sequence are changeable

SEQUENCE_LENGTH = 10
SEQUENCE_STRIDE = 10

BATCH_SIZE = 8
PREICTAL_PER_BATCH = 2
INTERICTAL_PER_BATCH = 6

TRAIN_BATCHES_TO_CACHE = 500
VAL_BATCHES_TO_CACHE = None

REBUILD_WINDOW_CACHE = False
REBUILD_SEQUENCE_CACHE = False

RUN_WINDOW_CACHE = False

SHUFFLE_TRAIN_BATCHES = True
SHUFFLE_VAL_BATCHES = False

RANDOM_SEED = 42

MAX_INTER_BUFFER = 5000
MAX_PRE_BUFFER = 5000

# HELPERS
# to CONVERT EEG TO MODEL SHAPE
def convert_to_model_shape(X):

      # the expected shape:
    # (windows, timepoints, channels)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D X, got {X.shape}")

        #from windows,channels,samples
        #to
        #windows.samples,channels

    if X.shape[1] == 17 and X.shape[2] == 1280:
        return np.transpose(X, (0, 2, 1))

    if X.shape[1] == 1280 and X.shape[2] == 17:
        return X

    raise ValueError(f"Unexpected X shape: {X.shape}")


def compute_freq_for_file(X_model):
    freq_list = []
# Processing every EEG window
    for i in range(X_model.shape[0]):

        # Extract frequency-domain features
        freq = compute_frequency_features(
            X_model[i],
            sfreq=TARGET_SFREQ
        )

        freq_list.append(freq)
# printing the Progress 
        if i % 500 == 0 or i == X_model.shape[0] - 1:
            print(f"    frequency [{i + 1}/{X_model.shape[0]}]")
 # Convert the list to numpy array
    return np.stack(freq_list).astype(np.float32)


def cache_window_files(file_paths, cache_dir, split_name):
    print("=" * 80)
    print(f"CACHING {split_name.upper()} WINDOW-LEVEL X + FREQ + Y")
    print("=" * 80)
 # Delete old cache if rebuilding
    if REBUILD_WINDOW_CACHE:
        for old_file in cache_dir.glob("*_window_freq.npz"):
            old_file.unlink()
        print(f"Old {split_name} window cache deleted.")

    saved = 0
    skipped = 0
  # Process every EEG file
    for file_i, file_path in enumerate(file_paths, start=1):
        file_path = Path(file_path)
         # Final cache file
        save_path = cache_dir / f"{file_path.stem}_window_freq.npz"
# Skip it if already cached 
        if save_path.exists():
            skipped += 1
            print(f"[{file_i}/{len(file_paths)}] already cached: {save_path.name}")
            continue

        print("=" * 80)
        print(f"[{file_i}/{len(file_paths)}] loading: {file_path.name}")

        X, y = load_prediction_npz_file(file_path)

        X_model = convert_to_model_shape(X).astype(np.float32)
        y = y.astype(np.int64)

        print("X:", X_model.shape)
        print("y:", y.shape)
        print("preictal:", int(np.sum(y == 1)))
        print("interictal:", int(np.sum(y == 0)))

        freq = compute_freq_for_file(X_model)

        np.savez(
            save_path,
            X=X_model,
            freq=freq,
            y=y,
            source_file=str(file_path)
        )

        saved += 1

        print("saved:", save_path.name)
        print("freq:", freq.shape)

    print("=" * 80)
    print(f"{split_name.upper()} WINDOW CACHE DONE")
    print("=" * 80)
    print("Saved new files:", saved)
    print("Skipped existing files:", skipped)


def build_sequences_from_window_cache(X, freq, y):
    """
    Memory-efficient generator.
    It gives one sequence at a time instead of storing all sequences in RAM.
    """

    max_start = len(y) - SEQUENCE_LENGTH

    if max_start < 0:
        return

    for start in range(0, max_start + 1, SEQUENCE_STRIDE):
        end = start + SEQUENCE_LENGTH

        X_seq = X[start:end]
        freq_seq = freq[start:end]

        y_seq = y[end - 1]

        yield X_seq, freq_seq, y_seq


def save_balanced_batch(
    cache_dir,
    prefix,
    batch_index,
    pre_X_buffer,
    pre_freq_buffer,
    pre_y_buffer,
    inter_X_buffer,
    inter_freq_buffer,
    inter_y_buffer,
    rng,
    shuffle_batches
):
    X_batch_list = (
        pre_X_buffer[:PREICTAL_PER_BATCH]
        + inter_X_buffer[:INTERICTAL_PER_BATCH]
    )

    freq_batch_list = (
        pre_freq_buffer[:PREICTAL_PER_BATCH]
        + inter_freq_buffer[:INTERICTAL_PER_BATCH]
    )

    y_batch_list = (
        pre_y_buffer[:PREICTAL_PER_BATCH]
        + inter_y_buffer[:INTERICTAL_PER_BATCH]
    )

    del pre_X_buffer[:PREICTAL_PER_BATCH]
    del pre_freq_buffer[:PREICTAL_PER_BATCH]
    del pre_y_buffer[:PREICTAL_PER_BATCH]

    del inter_X_buffer[:INTERICTAL_PER_BATCH]
    del inter_freq_buffer[:INTERICTAL_PER_BATCH]
    del inter_y_buffer[:INTERICTAL_PER_BATCH]

    X_batch = np.stack(X_batch_list).astype(np.float32)
    freq_batch = np.stack(freq_batch_list).astype(np.float32)
    y_batch = np.array(y_batch_list, dtype=np.int64)

    if shuffle_batches:
        order = rng.permutation(len(y_batch))
        X_batch = X_batch[order]
        freq_batch = freq_batch[order]
        y_batch = y_batch[order]

    preictal = int(np.sum(y_batch == 1))
    interictal = int(np.sum(y_batch == 0))

    if preictal != PREICTAL_PER_BATCH or interictal != INTERICTAL_PER_BATCH:
        raise ValueError(
            f"Bad balance in {prefix}_batch_{batch_index:05d}: "
            f"preictal={preictal}, interictal={interictal}"
        )

    if np.isnan(X_batch).any() or np.isnan(freq_batch).any():
        raise ValueError(f"NaN found in {prefix}_batch_{batch_index:05d}")

    if np.isinf(X_batch).any() or np.isinf(freq_batch).any():
        raise ValueError(f"Inf found in {prefix}_batch_{batch_index:05d}")

    save_path = cache_dir / f"{prefix}_batch_{batch_index:05d}.npz"
    temp_path = cache_dir / f"{prefix}_batch_{batch_index:05d}.tmp.npz"

    np.savez(
        temp_path,
        X=X_batch,
        freq=freq_batch,
        y=y_batch
    )

    temp_path.replace(save_path)

    return save_path, preictal, interictal


def build_balanced_sequence_batches_from_window_cache(
    window_cache_dir,
    sequence_cache_dir,
    prefix,
    batches_to_cache,
    shuffle_batches
):
    print("=" * 80)
    print(f"BUILDING {prefix.upper()} BALANCED LSTM SEQUENCE BATCHES")
    print("=" * 80)

    if REBUILD_SEQUENCE_CACHE:
        for old_file in sequence_cache_dir.glob(f"{prefix}_batch_*.npz"):
            old_file.unlink()
        for old_file in sequence_cache_dir.glob(f"{prefix}_batch_*.tmp.npz"):
            old_file.unlink()
        print(f"Old {prefix} sequence cache deleted.")

    for tmp_file in sequence_cache_dir.glob(f"{prefix}_batch_*.tmp.npz"):
        tmp_file.unlink()

    window_files = sorted(window_cache_dir.glob("*_window_freq.npz"))

    if len(window_files) == 0:
        raise ValueError(f"No window cache files found in {window_cache_dir}")

    existing_batches = sorted(sequence_cache_dir.glob(f"{prefix}_batch_*.npz"))
    start_index = len(existing_batches)

    print("Window cache files:", len(window_files))
    print("Existing sequence batches:", start_index)
    print("New batches to cache:", batches_to_cache if batches_to_cache is not None else "FULL POSSIBLE")
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Sequence stride:", SEQUENCE_STRIDE)
    print("Batch size:", BATCH_SIZE)
    print("Preictal per batch:", PREICTAL_PER_BATCH)
    print("Interictal per batch:", INTERICTAL_PER_BATCH)
    print("Shuffle batches:", shuffle_batches)
    print("Max inter buffer:", MAX_INTER_BUFFER)
    print("Max pre buffer:", MAX_PRE_BUFFER)

    rng = np.random.default_rng(RANDOM_SEED + start_index)

    pre_X_buffer = []
    pre_freq_buffer = []
    pre_y_buffer = []

    inter_X_buffer = []
    inter_freq_buffer = []
    inter_y_buffer = []

    saved_batches = 0
    batch_index = start_index

    total_preictal = 0
    total_interictal = 0

    for file_i, file_path in enumerate(window_files, start=1):

        if batches_to_cache is not None and saved_batches >= batches_to_cache:
            break

        print("=" * 80)
        print(f"[{prefix.upper()} WINDOW FILE {file_i}/{len(window_files)}] {file_path.name}")

        data = np.load(file_path, allow_pickle=True)

        X = data["X"].astype(np.float32)
        freq = data["freq"].astype(np.float32)
        y = data["y"].astype(np.int64)

        print("X:", X.shape)
        print("freq:", freq.shape)
        print("y:", y.shape)
        print("preictal:", int(np.sum(y == 1)))
        print("interictal:", int(np.sum(y == 0)))

        num_sequences = max(0, len(y) - SEQUENCE_LENGTH + 1)
        print("Sequences possible:", num_sequences)

        for X_seq, freq_seq, y_seq in build_sequences_from_window_cache(X, freq, y):

            if y_seq == 1:
                if len(pre_y_buffer) < MAX_PRE_BUFFER:
                    pre_X_buffer.append(X_seq)
                    pre_freq_buffer.append(freq_seq)
                    pre_y_buffer.append(y_seq)
            else:
                if len(inter_y_buffer) < MAX_INTER_BUFFER:
                    inter_X_buffer.append(X_seq)
                    inter_freq_buffer.append(freq_seq)
                    inter_y_buffer.append(y_seq)

            while (
                len(pre_y_buffer) >= PREICTAL_PER_BATCH
                and len(inter_y_buffer) >= INTERICTAL_PER_BATCH
            ):
                if batches_to_cache is not None and saved_batches >= batches_to_cache:
                    break

                save_path, preictal, interictal = save_balanced_batch(
                    cache_dir=sequence_cache_dir,
                    prefix=prefix,
                    batch_index=batch_index,
                    pre_X_buffer=pre_X_buffer,
                    pre_freq_buffer=pre_freq_buffer,
                    pre_y_buffer=pre_y_buffer,
                    inter_X_buffer=inter_X_buffer,
                    inter_freq_buffer=inter_freq_buffer,
                    inter_y_buffer=inter_y_buffer,
                    rng=rng,
                    shuffle_batches=shuffle_batches
                )

                total_preictal += preictal
                total_interictal += interictal

                saved_batches += 1
                batch_index += 1

                target = batches_to_cache if batches_to_cache is not None else "FULL"

                if saved_batches % 20 == 0 or saved_batches == 1:
                    print(
                        f"[{saved_batches}/{target}] cached {save_path.name} | "
                        f"preictal={preictal} | interictal={interictal} | "
                        f"buffer_preictal={len(pre_y_buffer)} | "
                        f"buffer_interictal={len(inter_y_buffer)}"
                    )

    print("=" * 80)
    print(f"{prefix.upper()} SEQUENCE CACHE DONE")
    print("=" * 80)
    print("New batches saved:", saved_batches)
    print("Total preictal:", total_preictal)
    print("Total interictal:", total_interictal)

    if total_preictal > 0:
        print("Interictal / Preictal ratio:", total_interictal / total_preictal)

    print("Remaining preictal buffer:", len(pre_y_buffer))
    print("Remaining interictal buffer:", len(inter_y_buffer))

# MAIN
def main():
    start_time = time.time()

    if PREICTAL_PER_BATCH + INTERICTAL_PER_BATCH != BATCH_SIZE:
        raise ValueError("PREICTAL_PER_BATCH + INTERICTAL_PER_BATCH must equal BATCH_SIZE")

    train_files, val_files, test_files = get_final17_prediction_file_lists()

    print("Train files:", len(train_files))
    print("Validation files:", len(val_files))
    print("Test files:", len(test_files))

    if RUN_WINDOW_CACHE:
        cache_window_files(
            file_paths=train_files,
            cache_dir=window_train_cache_dir,
            split_name="train"
        )

        cache_window_files(
            file_paths=val_files,
            cache_dir=window_val_cache_dir,
            split_name="val"
        )
    else:
        print("Skipping window cache because RUN_WINDOW_CACHE = False")

    build_balanced_sequence_batches_from_window_cache(
        window_cache_dir=window_train_cache_dir,
        sequence_cache_dir=sequence_train_cache_dir,
        prefix="train",
        batches_to_cache=TRAIN_BATCHES_TO_CACHE,
        shuffle_batches=SHUFFLE_TRAIN_BATCHES
    )

    build_balanced_sequence_batches_from_window_cache(
        window_cache_dir=window_val_cache_dir,
        sequence_cache_dir=sequence_val_cache_dir,
        prefix="val",
        batches_to_cache=VAL_BATCHES_TO_CACHE,
        shuffle_batches=SHUFFLE_VAL_BATCHES
    )

    print("=" * 80)
    print("ALL WINDOW + LSTM SEQUENCE CACHING DONE")
    print("=" * 80)
    print(f"Time: {(time.time() - start_time) / 60:.2f} min")


if __name__ == "__main__":
    main()