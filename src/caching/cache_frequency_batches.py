from pathlib import Path
import sys
import time
import numpy as np

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import compute_frequency_features

# SETTINGS

INPUT_CACHE_DIR = PROJECT_ROOT / "data" / "cached_train_batches"
OUTPUT_CACHE_DIR = PROJECT_ROOT / "data" / "cached_train_batches_with_freq_v3"

OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

batch_files = sorted(INPUT_CACHE_DIR.glob("train_batch_*.npz"))


print("FREQUENCY CACHE CREATION - TRAIN")

print("Input dir:", INPUT_CACHE_DIR)
print("Output dir:", OUTPUT_CACHE_DIR)
print("Input batches:", len(batch_files))


if len(batch_files) == 0:
    raise ValueError("No training batches found. Check INPUT_CACHE_DIR.")


start_time = time.time()

for i, batch_file in enumerate(batch_files, start=1):

    save_path = OUTPUT_CACHE_DIR / batch_file.name

    if save_path.exists():
        if i % 50 == 0:
            print(f"[{i}/{len(batch_files)}] skipped existing {save_path.name}")
        continue

    data = np.load(batch_file)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    freq_batch = []

    for j in range(len(X)):
        freq = compute_frequency_features(X[j])
        freq_batch.append(freq)

    freq_batch = np.array(freq_batch, dtype=np.float32)

    np.savez(
        save_path,
        X=X,
        freq=freq_batch,
        y=y
    )

    if i % 20 == 0 or i == 1 or i == len(batch_files):
        print(
            f"[{i}/{len(batch_files)}] cached {batch_file.name} | "
            f"X={X.shape} | freq={freq_batch.shape} | y={y.shape}"
        )


print("DONE")
print(f"Time: {(time.time() - start_time) / 60:.2f} min")
