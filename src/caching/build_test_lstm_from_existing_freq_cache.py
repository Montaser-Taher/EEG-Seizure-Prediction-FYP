from pathlib import Path
import sys
import json
import shutil
import numpy as np

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# ============================================================
# SETTINGS
# ============================================================

INPUT_CACHE_DIR = PROJECT_ROOT / "data" / "cached_test_batches_with_freq_v3"

# SAVE OUTPUT TO EXTERNAL DRIVE
OUTPUT_CACHE_DIR = Path("/mnt/d/EEG_FYP_DATA/data/cached_test_lstm_seq5_batches_with_freq_fast")

SEQUENCE_LENGTH = 5
BATCH_SIZE = 32

TIMEPOINTS = 1280
CHANNELS = 17
FREQ_FEATURES = 323

# IMPORTANT:
# False = resume / continue
# True = delete old output and start again
REBUILD_FROM_ZERO = False

# ============================================================
# RESET OR CREATE OUTPUT
# ============================================================

if REBUILD_FROM_ZERO and OUTPUT_CACHE_DIR.exists():
    shutil.rmtree(OUTPUT_CACHE_DIR)

OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

batch_files = sorted(INPUT_CACHE_DIR.glob("test_batch_*.npz"))

if len(batch_files) == 0:
    raise ValueError(f"No test batches found in {INPUT_CACHE_DIR}")

# Resume from existing output batches
existing_output_batches = sorted(OUTPUT_CACHE_DIR.glob("test_batch_*.npz"))
batch_index = len(existing_output_batches)

target_start_sequence = batch_index * BATCH_SIZE

print("=" * 80)
print("STREAMING CNN-LSTM TEST SEQUENCE CACHE")
print("=" * 80)
print("Input:", INPUT_CACHE_DIR)
print("Output:", OUTPUT_CACHE_DIR)
print("Input batches:", len(batch_files))
print("Existing output batches:", len(existing_output_batches))
print("Next batch index:", batch_index)
print("Skipping already created sequences:", target_start_sequence)
print("=" * 80)

# ============================================================
# VARIABLES
# ============================================================

sequence_X = []
sequence_freq = []
sequence_y = []

save_X = []
save_freq = []
save_y = []

seen_sequences = 0

total_sequences = batch_index * BATCH_SIZE
total_preictal = 0
total_interictal = 0


def save_sequence_batch():
    global save_X, save_freq, save_y
    global batch_index, total_sequences, total_preictal, total_interictal

    X_batch = np.array(save_X, dtype=np.float32)
    freq_batch = np.array(save_freq, dtype=np.float32)
    y_batch = np.array(save_y, dtype=np.int64)

    save_path = OUTPUT_CACHE_DIR / f"test_batch_{batch_index:05d}.npz"

    temp_path = save_path.with_suffix(".tmp.npz")

    np.savez_compressed(
        temp_path,
        X=X_batch,
        freq=freq_batch,
        y=y_batch
    )

    temp_path.replace(save_path)

    preictal_count = int(np.sum(y_batch == 1))
    interictal_count = int(np.sum(y_batch == 0))

    total_sequences += len(y_batch)
    total_preictal += preictal_count
    total_interictal += interictal_count

    print(
        f"Saved {save_path.name} | "
        f"X={X_batch.shape} | "
        f"freq={freq_batch.shape} | "
        f"preictal={preictal_count} | "
        f"interictal={interictal_count}"
    )

    batch_index += 1
    save_X = []
    save_freq = []
    save_y = []


# Delete broken temp files if previous run crashed
for tmp_file in OUTPUT_CACHE_DIR.glob("test_batch_*.tmp.npz"):
    tmp_file.unlink()

# ============================================================
# STREAM THROUGH EXISTING TEST BATCHES
# ============================================================

for file_i, batch_file in enumerate(batch_files, start=1):

    data = np.load(batch_file)

    X = data["X"].astype(np.float32)
    freq = data["freq"].astype(np.float32)
    y = data["y"].astype(np.int64)

    if X.shape[1:] != (TIMEPOINTS, CHANNELS):
        raise ValueError(f"Bad X shape in {batch_file.name}: {X.shape}")

    if freq.shape[1] != FREQ_FEATURES:
        raise ValueError(f"Bad freq shape in {batch_file.name}: {freq.shape}")

    for i in range(len(y)):

        sequence_X.append(X[i])
        sequence_freq.append(freq[i])
        sequence_y.append(y[i])

        if len(sequence_X) == SEQUENCE_LENGTH:

            # Count this sequence
            seen_sequences += 1

            # Resume logic: skip sequences already saved
            if seen_sequences <= target_start_sequence:
                sequence_X.pop(0)
                sequence_freq.pop(0)
                sequence_y.pop(0)
                continue

            X_seq = np.array(sequence_X, dtype=np.float32)
            freq_seq = np.array(sequence_freq, dtype=np.float32)

            # label = last window label
            y_seq = sequence_y[-1]

            save_X.append(X_seq)
            save_freq.append(freq_seq)
            save_y.append(y_seq)

            # slide by 1 window
            sequence_X.pop(0)
            sequence_freq.pop(0)
            sequence_y.pop(0)

            if len(save_X) == BATCH_SIZE:
                save_sequence_batch()

    if file_i % 100 == 0 or file_i == len(batch_files):
        print(f"Processed input batch [{file_i}/{len(batch_files)}]")

# Save remaining sequences
if len(save_X) > 0:
    save_sequence_batch()

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "input_cache_dir": str(INPUT_CACHE_DIR),
    "output_cache_dir": str(OUTPUT_CACHE_DIR),
    "sequence_length": SEQUENCE_LENGTH,
    "batch_size": BATCH_SIZE,
    "label_strategy": "last window label",
    "shuffle": False,
    "balanced": False,
    "resume_enabled": True,
    "existing_batches_before_run": len(existing_output_batches),
    "final_batch_index": batch_index,
    "note": "Streaming version. Saves to external drive and resumes from existing output batches."
}

with open(OUTPUT_CACHE_DIR / "test_lstm_seq5_cache_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("=" * 80)
print("DONE")
print("=" * 80)
print("Final output batches:", batch_index)
print("Saved to:", OUTPUT_CACHE_DIR)