from pathlib import Path
import sys
import numpy as np
import time

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import (
    get_final17_prediction_file_lists,
    load_prediction_npz_file
)

cache_dir = PROJECT_ROOT / "data" / "cached_val_batches"
cache_dir.mkdir(parents=True, exist_ok=True)

# SETTINGS
batch_size = 32
preictal_per_batch = 8
interictal_per_batch = 24

# Set to None for full validation cache
# Set to 100 first if you want a small test
num_batches_to_add = None

# If True, deletes old validation batches and starts again
rebuild_from_zero = False

# Makeing sure batch balance is correct
if preictal_per_batch + interictal_per_batch != batch_size:
    raise ValueError("preictal_per_batch + interictal_per_batch must equal batch_size")

train_files, val_files, test_files = get_final17_prediction_file_lists()

# Deleting old validation batches if rebuild enabled
if rebuild_from_zero:
    for old_file in cache_dir.glob("val_batch_*.npz"):
        old_file.unlink()
    print("Old validation cache deleted.")

existing_batches = sorted(cache_dir.glob("val_batch_*.npz"))
# Starting batch index
# Allows cache to continue from where it stopped before
start_index = len(existing_batches)
# Calculate how many windows to skip when resuming caching
preictal_to_skip = start_index * preictal_per_batch
interictal_to_skip = start_index * interictal_per_batch


print("VALIDATION CACHE SETTINGS")
print("Validation files:", len(val_files))
print("Existing cached validation batches:", start_index)
print("New batches to add:", num_batches_to_add if num_batches_to_add is not None else "FULL POSSIBLE")
print("Batch size:", batch_size)
print("Preictal per batch:", preictal_per_batch)
print("Interictal per batch:", interictal_per_batch)
print("Resume skip preictal windows:", preictal_to_skip)
print("Resume skip interictal windows:", interictal_to_skip)

# Start timer
start_time = time.time()
# BUFFERS
pre_buffer = []
pre_y_buffer = []

inter_buffer = []
inter_y_buffer = []
# Final counters
new_preictal_total = 0
new_interictal_total = 0
bad_new_batches = 0
saved_new_batches = 0

# HELPER FUNCTION
# Counts total windows inside buffer
def count_buffer(buffer):
    return sum(arr.shape[0] for arr in buffer)

# HELPER FUNCTION
# Takes windows from buffer
def take_from_buffer(X_buffer, y_buffer, n):
    X_parts = []
    y_parts = []
    remaining = n

    while remaining > 0 and len(X_buffer) > 0:
        X_first = X_buffer[0]
        y_first = y_buffer[0]

        take = min(remaining, X_first.shape[0])

        X_parts.append(X_first[:take])
        y_parts.append(y_first[:take])
  # Removing used arrays from buffer
        if take == X_first.shape[0]:
            X_buffer.pop(0)
            y_buffer.pop(0)
        else:
            X_buffer[0] = X_first[take:]
            y_buffer[0] = y_first[take:]

        remaining -= take
 # Safety check
    if remaining != 0:
        raise ValueError("Not enough windows in buffer.")

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def save_one_batch(batch_index, rng):
    global new_preictal_total, new_interictal_total, bad_new_batches

    X_pre, y_pre = take_from_buffer(pre_buffer, pre_y_buffer, preictal_per_batch)
    X_inter, y_inter = take_from_buffer(inter_buffer, inter_y_buffer, interictal_per_batch)

    X_batch = np.concatenate([X_pre, X_inter], axis=0)
    y_batch = np.concatenate([y_pre, y_inter], axis=0)

    order = rng.permutation(len(y_batch))
    X_batch = X_batch[order]
    y_batch = y_batch[order]

    # Convert from (batch, channels, samples) to (batch, samples, channels)
    if X_batch.ndim == 3 and X_batch.shape[1] == 17:
        X_batch = np.transpose(X_batch, (0, 2, 1))

    preictal_count = int(np.sum(y_batch == 1))
    interictal_count = int(np.sum(y_batch == 0))

    new_preictal_total += preictal_count
    new_interictal_total += interictal_count

    if preictal_count != preictal_per_batch or interictal_count != interictal_per_batch:
        bad_new_batches += 1
        print(
            f"WARNING bad balance in val_batch_{batch_index:05d}: "
            f"preictal={preictal_count}, interictal={interictal_count}"
        )

    if np.isnan(X_batch).any():
        print(f"WARNING NaN found before saving val_batch_{batch_index:05d}")

    if np.isinf(X_batch).any():
        print(f"WARNING Inf found before saving val_batch_{batch_index:05d}")

    save_path = cache_dir / f"val_batch_{batch_index:05d}.npz"

    np.savez(
        save_path,
        X=X_batch.astype(np.float32),
        y=y_batch.astype(np.int64)
    )

    return preictal_count, interictal_count


rng = np.random.default_rng(123 + start_index)

print("Creating cached validation batches...")


stop_caching = False

for file_i, file_path in enumerate(val_files, start=1):

    if stop_caching:
        break

    print(f"[FILE {file_i}/{len(val_files)}] Loading: {Path(file_path).name}")

    X, y = load_prediction_npz_file(file_path)

    pre_idx = np.where(y == 1)[0]
    inter_idx = np.where(y == 0)[0]

    print(
        f"Loaded {Path(file_path).name} | "
        f"windows={len(y)} | "
        f"preictal={len(pre_idx)} | "
        f"interictal={len(inter_idx)} | "
        f"X shape={X.shape}"
    )

    # Skip already cached preictal windows when resuming
    if preictal_to_skip > 0:
        skip_now = min(preictal_to_skip, len(pre_idx))
        pre_idx = pre_idx[skip_now:]
        preictal_to_skip -= skip_now

    # Skip already cached interictal windows when resuming
    if interictal_to_skip > 0:
        skip_now = min(interictal_to_skip, len(inter_idx))
        inter_idx = inter_idx[skip_now:]
        interictal_to_skip -= skip_now

    if len(pre_idx) > 0:
        pre_buffer.append(X[pre_idx])
        pre_y_buffer.append(y[pre_idx])

    if len(inter_idx) > 0:
        inter_buffer.append(X[inter_idx])
        inter_y_buffer.append(y[inter_idx])

    print(
        f"Current buffer | "
        f"preictal={count_buffer(pre_buffer)} | "
        f"interictal={count_buffer(inter_buffer)}"
    )

    while (
        count_buffer(pre_buffer) >= preictal_per_batch
        and count_buffer(inter_buffer) >= interictal_per_batch
    ):
        if num_batches_to_add is not None and saved_new_batches >= num_batches_to_add:
            stop_caching = True
            break

        batch_index = start_index + saved_new_batches

        preictal_count, interictal_count = save_one_batch(batch_index, rng)

        saved_new_batches += 1

        if saved_new_batches % 20 == 0 or saved_new_batches == 1:
            print(
                f"[{saved_new_batches}/{num_batches_to_add if num_batches_to_add is not None else 'FULL'}] "
                f"saved val_batch_{batch_index:05d} | "
                f"preictal={preictal_count} | "
                f"interictal={interictal_count} | "
                f"buffer_preictal={count_buffer(pre_buffer)} | "
                f"buffer_interictal={count_buffer(inter_buffer)}"
            )


print("VALIDATION CACHING DONE")

print("New validation batches saved:", saved_new_batches)
print("New preictal windows:", new_preictal_total)
print("New interictal windows:", new_interictal_total)
print("New total windows:", new_preictal_total + new_interictal_total)
print("Bad new batches:", bad_new_batches)
print(f"Time: {(time.time() - start_time) / 60:.2f} min")


print("\nChecking all cached validation batches...")

batch_files = sorted(cache_dir.glob("val_batch_*.npz"))

total_preictal = 0
total_interictal = 0
bad_batches = 0
nan_batches = 0
inf_batches = 0

for i, batch_file in enumerate(batch_files):
    data = np.load(batch_file)

    X = data["X"]
    y = data["y"]

    preictal = int(np.sum(y == 1))
    interictal = int(np.sum(y == 0))

    total_preictal += preictal
    total_interictal += interictal

    if preictal != preictal_per_batch or interictal != interictal_per_batch:
        bad_batches += 1
        print(f"Bad balance: {batch_file.name} | preictal={preictal}, interictal={interictal}")

    if np.isnan(X).any():
        nan_batches += 1
        print(f"NaN found in: {batch_file.name}")

    if np.isinf(X).any():
        inf_batches += 1
        print(f"Inf found in: {batch_file.name}")

    if i == 0:
        print("Example X shape:", X.shape)
        print("Example y shape:", y.shape)
        print("Example range:", X.min(), X.max())


print("FINAL VALIDATION CACHE CHECK")

print("Total cached validation batches:", len(batch_files))
print("Total cached validation windows:", total_preictal + total_interictal)
print("Total preictal:", total_preictal)
print("Total interictal:", total_interictal)
print("Bad balance batches:", bad_batches)
print("NaN batches:", nan_batches)
print("Inf batches:", inf_batches)

if total_preictal > 0:
    print("Interictal / Preictal ratio:", total_interictal / total_preictal)

