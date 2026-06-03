from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path("../src").resolve()))

from config import (
    RAW_ROOT,
    PROCESSED_GROUP2,
    GROUP2_PATIENTS,
    CHANNEL_ORDER,
    NOTCH_FREQ,
    LOW_FREQ,
    HIGH_FREQ,
    WINDOW_SIZE_SEC,
    STEP_SEC,
    BAD_CHANNEL_Z_THRESH
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

# to store all patients summaries together
group2_all_summary_rows = []

print("Starting Group 2 preprocessing...")
print(f"Patients in Group 2: {GROUP2_PATIENTS}")


# BUILD ONE GLOBAL COMMON CHANNEL SET FOR ALL GROUP 2


group2_all_edf_files = []

for patient_name in ["chb20"]:
    patient_folder = RAW_ROOT / patient_name
    edf_files = sorted(patient_folder.glob("*.edf"))
    group2_all_edf_files.extend(edf_files)

print(f"\nTotal EDF files across all Group 2 patients: {len(group2_all_edf_files)}")

if len(group2_all_edf_files) == 0:
    raise ValueError("No EDF files found across Group 2 patients.")

# Start from first file as reference
first_raw = load_raw_edf(group2_all_edf_files[0])
first_raw = prepare_raw(first_raw)
reference_channels = first_raw.ch_names.copy()

group2_global_common = set(reference_channels)

# Find channels common across ALL files of ALL Group 2 patients
for file_path in group2_all_edf_files[1:]:
    raw_tmp = load_raw_edf(file_path)
    raw_tmp = prepare_raw(raw_tmp)
    current_channels = set(raw_tmp.ch_names)
    group2_global_common = group2_global_common.intersection(current_channels)

# Keeping the order as close as possible to Group 1 / standard CHANNEL_ORDER
expected_channels = [ch for ch in CHANNEL_ORDER if ch in group2_global_common]

# Add any extra channels that are common but not listed in CHANNEL_ORDER
extra_common_channels = [ch for ch in reference_channels if ch in group2_global_common and ch not in expected_channels]
expected_channels.extend(extra_common_channels)

print("\nGlobal common channels across ALL Group 2 patients:")
print(expected_channels)
print(f"Number of global common channels: {len(expected_channels)}")

# LOOP THROUGH ALL GROUP 2 PATIENTS

for patient_name in ["chb20"]:
    print("\n" + "=" * 80)
    print(f"Processing patient: {patient_name}")

    patient_folder = RAW_ROOT / patient_name
    summary_path = patient_folder / f"{patient_name}-summary.txt"

    # output folder for this patient
    output_folder = PROCESSED_GROUP2 / patient_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # get all EDF files in this patient folder
    edf_files = sorted(patient_folder.glob("*.edf"))

    print(f"Found {len(edf_files)} EDF files")

    if len(edf_files) == 0:
        print(f"No EDF files found for {patient_name}, skipping...")
        continue

    # Read seizure information from summary file once
    summary_dict = parse_chbmit_summary(summary_path)

    print("\nUsing the same global Group 2 common channels for this patient:")
    print(expected_channels)
    print(f"Number of expected channels: {len(expected_channels)}")

    # summary rows for this patient only
    patient_summary_rows = []

   
    # LOOP THROUGH ALL FILES OF THIS PATIENT
   
    for file_path in edf_files:
        file_name = file_path.name
        seizure_intervals = summary_dict.get(file_name, [])

        print("\n" + "-" * 60)
        print(f"Processing file: {file_name}")
        print(f"Seizure intervals: {seizure_intervals}")

        # Load raw EDF file
        raw = load_raw_edf(file_path)
        print("Loaded raw file successfully")

        # Keep EEG channels only and set them as EEG type
        raw = prepare_raw(raw)

        print("EEG channels kept before reorder:")
        print(raw.ch_names)
        print(f"Number of EEG channels before reorder: {len(raw.ch_names)}")

        # Reorder channels to ensure consistency
        raw = reorder_channels(raw, expected_channels)

        # save channel names if needed
        ch_names = raw.ch_names.copy()

        print("EEG channels after reorder/common-channel selection:")
        print(raw.ch_names)
        print(f"Number of EEG channels after reorder: {len(raw.ch_names)}")

        # Check sampling rate
        print(f"Original sampling rate: {raw.info['sfreq']} Hz")

        # Apply filtering
        raw_before_filter = raw.copy()

        raw = apply_filters(
            raw,
            notch_freq=NOTCH_FREQ,
            l_freq=LOW_FREQ,
            h_freq=HIGH_FREQ
        )

        raw_after_filter = raw.copy()

        print(f"Filtering done: notch {NOTCH_FREQ} Hz + band-pass {LOW_FREQ}-{HIGH_FREQ} Hz")

        # Detect suspicious bad channels
        bad_channels, channel_std = detect_bad_channels(raw, z_thresh=BAD_CHANNEL_Z_THRESH)
        raw.info["bads"] = bad_channels

        print(f"Suspicious bad channels: {bad_channels}")

        # Convert EEG signal to NumPy array
        data_before_norm = raw.get_data()
        sfreq = raw.info["sfreq"]

        print(f"Data shape before normalization: {data_before_norm.shape}")
        print(f"Sampling frequency used: {sfreq}")

        # Normalize
        data_after_norm = zscore_normalize(data_before_norm.copy())
        data = data_after_norm

        print("Normalization done")

        # Split into windows and label them
        X, y = create_windows_and_labels(
            data=data,
            sfreq=sfreq,
            window_size_sec=WINDOW_SIZE_SEC,
            step_sec=STEP_SEC,
            seizure_intervals=seizure_intervals
        )

        print(f"Windowed data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of seizure windows: {np.sum(y)}")
        print(f"Number of non-seizure windows: {len(y) - np.sum(y)}")

        # Save processed result for this file
        save_path = output_folder / f"{Path(file_name).stem}_processed.npz"
        save_processed_data(save_path, X, y)

        print(f"Preprocessing finished successfully for {file_name}")

        # summary row for this file
        row = {
            "patient_name": patient_name,
            "file_name": file_name,
            "n_channels_after_common_selection": len(raw.ch_names),
            "sampling_rate": raw.info["sfreq"],
            "duration_sec": raw.n_times / raw.info["sfreq"],
            "n_seizure_intervals": len(seizure_intervals),
            "n_bad_channels": len(bad_channels),
            "n_windows": len(y),
            "n_seizure_windows": int(np.sum(y == 1)),
            "n_nonseizure_windows": int(np.sum(y == 0))
        }

        patient_summary_rows.append(row)
        group2_all_summary_rows.append(row)

    print(f"\nAll files in {patient_name} processed successfully.")


    # SAVE ONE SUMMARY FOR the PATIENT
  
    df_patient_summary = pd.DataFrame(patient_summary_rows)

    patient_summary_save_path = output_folder / f"{patient_name}_processing_summary.csv"
    df_patient_summary.to_csv(patient_summary_save_path, index=False)

    print(f"\nPatient summary saved to: {patient_summary_save_path}")
    print(df_patient_summary.head())


# SAVE ONE COMBINED SUMMARY FOR ALL GROUP 2

df_group2_summary = pd.DataFrame(group2_all_summary_rows)

group2_summary_save_path = PROCESSED_GROUP2 / "group2_all_patients_summary.csv"
df_group2_summary.to_csv(group2_summary_save_path, index=False)

print("\n" + "=" * 80)
print("All Group 2 patients processed successfully.")
print(f"Combined Group 2 summary saved to: {group2_summary_save_path}")
print(df_group2_summary.head())