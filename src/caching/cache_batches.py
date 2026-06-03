from pathlib import Path
import sys
import json
import time
import numpy as np

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import (
    get_final17_prediction_file_lists,
    prediction_batch_generator,
    transpose_generator
)

results_dir = PROJECT_ROOT / "results"
cache_dir = PROJECT_ROOT / "data" / "cached_train_batches"
cache_dir.mkdir(parents=True, exist_ok=True)

# SETTINGS

batch_size = 20
num_batches_to_cache = 2600
preictal_fraction = 0.20

expected_preictal_per_batch = int(batch_size * preictal_fraction)
expected_interictal_per_batch = batch_size - expected_preictal_per_batch


# LOAD DATA + PLAN

train_files, val_files, test_files = get_final17_prediction_file_lists()
print("Train files:", len(train_files))

with open(results_dir / "balance_plan_train_ratio_4.json", "r") as f:
    balance_plan = json.load(f)

print("Loaded balance plan:")
print(balance_plan)

# RESUME LOGIC

existing_batches = sorted(cache_dir.glob("train_batch_*.npz"))
start_index = len(existing_batches)


print("CACHE SETTINGS")

print("Existing cached batches:", start_index)
print("New batches to add:", num_batches_to_cache)
print("Batch size:", batch_size)
print("Expected per batch:")
print("Preictal:", expected_preictal_per_batch)
print("Interictal:", expected_interictal_per_batch)



# GENERATOR

train_generator = prediction_batch_generator(
    file_paths=train_files,
    batch_size=batch_size,
    balance_plan=balance_plan,
    preictal_fraction=preictal_fraction,
    shuffle=True,
    random_state=42 + start_index
)

train_generator = transpose_generator(train_generator)


# CACHE BATCHES

print("Creating cached batches...")
start_time = time.time()

new_preictal_total = 0
new_interictal_total = 0
bad_new_batches = 0

for i in range(num_batches_to_cache):
    batch_index = start_index + i

    X_batch, y_batch = next(train_generator)

    preictal_count = int(np.sum(y_batch == 1))
    interictal_count = int(np.sum(y_batch == 0))

    new_preictal_total += preictal_count
    new_interictal_total += interictal_count

    if preictal_count != expected_preictal_per_batch or interictal_count != expected_interictal_per_batch:
        bad_new_batches += 1
        print(
            f"WARNING bad balance in batch {batch_index:05d}: "
            f"preictal={preictal_count}, interictal={interictal_count}"
        )

    save_path = cache_dir / f"train_batch_{batch_index:05d}.npz"

    np.savez(
        save_path,
        X=X_batch.astype(np.float32),
        y=y_batch.astype(np.int64)
    )

    if i % 20 == 0 or i == num_batches_to_cache - 1:
        print(
            f"[{i + 1}/{num_batches_to_cache}] saved "
            f"batch {batch_index:05d} | "
            f"preictal={preictal_count} | "
            f"interictal={interictal_count}"
        )


print("CACHING DONE")

print("New preictal windows:", new_preictal_total)
print("New interictal windows:", new_interictal_total)
print("New total windows:", new_preictal_total + new_interictal_total)
print("Bad new batches:", bad_new_batches)
print(f"Time: {(time.time() - start_time) / 60:.2f} min")


# CHECK ALL CACHED BATCHES

print("\nChecking all cached batches...")

batch_files = sorted(cache_dir.glob("train_batch_*.npz"))

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

    if preictal != expected_preictal_per_batch or interictal != expected_interictal_per_batch:
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


print("FINAL CACHE CHECK")

print("Total cached batches:", len(batch_files))
print("Total cached windows:", total_preictal + total_interictal)
print("Total preictal:", total_preictal)
print("Total interictal:", total_interictal)
print("Bad balance batches:", bad_batches)
print("NaN batches:", nan_batches)
print("Inf batches:", inf_batches)

if total_preictal > 0:
    print("Interictal / Preictal ratio:", total_interictal / total_preictal)

