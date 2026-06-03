from pathlib import Path
import numpy as np
import pandas as pd
import sys

# I set the main project path here so the script can find the src folder and project files.
PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

# This allows Python to import config.py and utils.py from the src folder.
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (
    RAW_ROOT,
    PROCESSED_GROUP1,
    NOTCH_FREQ,
    LOW_FREQ,
    HIGH_FREQ,
    WINDOW_SIZE_SEC,
    STEP_SEC,
    BAD_CHANNEL_Z_THRESH,
    FINAL_17_CHANNELS
)

from utils import (
    load_raw_edf,
    prepare_raw,
    apply_filters,
    parse_chbmit_summary,
    detect_bad_channels,
    reorder_channels,
    zscore_normalize,
    create_windows_and_labels,
    save_processed_data
)

# I process only this patient here. This is useful when testing preprocessing on one patient first.
PATIENTS_TO_PROCESS = ["chb03"]

# This list stores summary information for all processed files.
group1_all_summary_rows = []

print("=" * 80)
print("Starting Group 1 FINAL_17 preprocessing")
print("Patients to process:", PATIENTS_TO_PROCESS)
print("=" * 80)

# These are the final 17 EEG channels used by the model.
# Keeping the same channels across files makes the model input consistent.
expected_channels = FINAL_17_CHANNELS

print("\nUsing FINAL_17 channels:")
print(expected_channels)
print("Number of expected channels:", len(expected_channels))

for patient_name in PATIENTS_TO_PROCESS:
    print("\n" + "=" * 80)
    print(f"Processing patient: {patient_name}")

    # Define the raw patient folder and summary file path.
    patient_folder = RAW_ROOT / patient_name
    summary_path = patient_folder / f"{patient_name}-summary.txt"

    # Create the output folder where processed .npz files will be saved.
    output_folder = PROCESSED_GROUP1 / patient_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get all EDF files for this patient.
    edf_files = sorted(patient_folder.glob("*.edf"))

    print(f"Found {len(edf_files)} EDF files")

    # If no EDF files exist, the patient is skipped.
    if len(edf_files) == 0:
        print(f"No EDF files found for {patient_name}, skipping...")
        continue

    # Read seizure start/end times from the CHB-MIT summary text file.
    summary_dict = parse_chbmit_summary(summary_path)

    # This stores summary rows for this patient only.
    patient_summary_rows = []

    for file_path in edf_files:
        file_name = file_path.name

        # Get seizure intervals for the current EDF file.
        # If the file has no seizure, this will be an empty list.
        seizure_intervals = summary_dict.get(file_name, [])

        print("\n" + "-" * 60)
        print(f"Processing file: {file_name}")
        print(f"Seizure intervals: {seizure_intervals}")

        # Output path for this processed file.
        save_path = output_folder / f"{Path(file_name).stem}_processed.npz"

        # This avoids repeating preprocessing if the file was already processed before.
        if save_path.exists():
            print(f"Skipping {file_name} because it is already preprocessed.")
            continue

        # Load the raw EDF file using MNE.
        raw = load_raw_edf(file_path)
        print("Loaded raw file successfully")

        # Keep EEG channels only and prepare the channel types.
        raw = prepare_raw(raw)

        print("EEG channels kept:")
        print(raw.ch_names)
        print(f"Number of EEG channels before FINAL_17 selection: {len(raw.ch_names)}")

        # Check whether all required final 17 channels exist in this file.
        missing_channels = [ch for ch in expected_channels if ch not in raw.ch_names]

        # If any required channel is missing, the file cannot be used for the final model.
        if len(missing_channels) > 0:
            print("Missing FINAL_17 channels:")
            print(missing_channels)
            print(f"Skipping {file_name} because it cannot be converted to FINAL_17.")
            continue

        # Reorder the channels into the exact same order used by the model.
        raw = reorder_channels(raw, expected_channels)

        print("EEG channels after FINAL_17 selection:")
        print(raw.ch_names)
        print(f"Number of EEG channels after FINAL_17 selection: {len(raw.ch_names)}")

        print(f"Original sampling rate: {raw.info['sfreq']} Hz")

        # Apply notch and band-pass filtering.
        # Notch removes powerline noise, and band-pass keeps useful EEG frequencies.
        raw = apply_filters(
            raw,
            notch_freq=NOTCH_FREQ,
            l_freq=LOW_FREQ,
            h_freq=HIGH_FREQ
        )

        print(f"Filtering done: notch {NOTCH_FREQ} Hz + band-pass {LOW_FREQ}-{HIGH_FREQ} Hz")

        # Detect suspicious channels using channel standard deviation.
        # I keep this mainly for quality checking and reporting.
        bad_channels, channel_std = detect_bad_channels(raw, z_thresh=BAD_CHANNEL_Z_THRESH)
        raw.info["bads"] = bad_channels

        print(f"Suspicious bad channels: {bad_channels}")

        # Get EEG data as a NumPy array.
        # Shape should be channels x samples.
        data_before_norm = raw.get_data()
        sfreq = raw.info["sfreq"]

        print(f"Data shape before normalization: {data_before_norm.shape}")
        print(f"Sampling frequency used: {sfreq}")

        # Apply z score normalization per channel.
        # This makes channel values more comparable before windowing.
        data_after_norm = zscore_normalize(data_before_norm.copy())
        data = data_after_norm

        print("Normalization done")

        # Split the continuous EEG signal into fixed-size windows.
        # Labels are created based on whether each window overlaps a seizure interval.
        X, y = create_windows_and_labels(
            data=data,
            sfreq=sfreq,
            window_size_sec=WINDOW_SIZE_SEC,
            step_sec=STEP_SEC,
            seizure_intervals=seizure_intervals
        )

        print(f"Windowed data shape: {X.shape}")

        # Safety check to make sure the model input has exactly 17 channels.
        if X.shape[1] != 17:
            raise ValueError(f"Expected 17 channels, got X shape {X.shape}")

        print(f"Labels shape: {y.shape}")
        print(f"Number of seizure windows: {np.sum(y)}")
        print(f"Number of non-seizure windows: {len(y) - np.sum(y)}")

        save_processed_data(save_path, X, y)

        print(f"Saved FINAL_17 processed file to: {save_path}")
        print(f"Preprocessing finished successfully for {file_name}")

        # Store file-level information so I can later check what was processed.
        row = {
            "patient_name": patient_name,
            "file_name": file_name,
            "n_channels": len(raw.ch_names),
            "sampling_rate": raw.info["sfreq"],
            "duration_sec": raw.n_times / raw.info["sfreq"],
            "n_seizure_intervals": len(seizure_intervals),
            "n_bad_channels": len(bad_channels),
            "n_windows": len(y),
            "n_seizure_windows": int(np.sum(y == 1)),
            "n_nonseizure_windows": int(np.sum(y == 0))
        }

        patient_summary_rows.append(row)
        group1_all_summary_rows.append(row)

    print(f"\nAll files in {patient_name} processed successfully.")

    df_patient_summary = pd.DataFrame(patient_summary_rows)

    patient_summary_save_path = output_folder / f"{patient_name}_final17_processing_summary.csv"
    df_patient_summary.to_csv(patient_summary_save_path, index=False)

    print(f"\nPatient summary saved to: {patient_summary_save_path}")
    print(df_patient_summary.head())

df_group1_summary = pd.DataFrame(group1_all_summary_rows)

group1_summary_save_path = PROCESSED_GROUP1 / "group1_chb03_final17_summary.csv"
df_group1_summary.to_csv(group1_summary_save_path, index=False)

print("\n" + "=" * 80)
print("Group 1 chb03 FINAL_17 preprocessing finished.")
print(f"Summary saved to: {group1_summary_save_path}")
print(df_group1_summary.head())