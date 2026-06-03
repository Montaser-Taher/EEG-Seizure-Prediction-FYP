from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path("../src").resolve()))

from config import (
    RAW_ROOT,
    PROCESSED_GROUP3,
    GROUP3_PATIENTS,
    BAD_FILES,
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
group3_all_summary_rows = []

# to store skipped files and why they were skipped
group3_skipped_rows = []

print("Starting Group 3 preprocessing...")
print(f"Patients in Group 3: {GROUP3_PATIENTS}")


# BUILD ONE GLOBAL COMMON CHANNEL SET FOR ALL VALID GROUP 3 FILES


group3_valid_edf_files = []

for patient_name in GROUP3_PATIENTS:
    patient_folder = RAW_ROOT / patient_name
    edf_files = sorted(patient_folder.glob("*.edf"))

    # convert to set for safer/faster checking
    patient_bad_files = set(BAD_FILES.get(patient_name, []))

    print("\n" + "=" * 80)
    print(f"Checking patient: {patient_name}")
    print(f"Found {len(edf_files)} EDF files")
    print(f"Known bad files: {sorted(patient_bad_files)}")

    for file_path in edf_files:
        file_name = file_path.name.strip()

        if file_name in patient_bad_files:
            print(f"Skipping known bad file: {file_name}")
            group3_skipped_rows.append({
                "patient_name": patient_name,
                "file_name": file_name,
                "reason": "known bad file from BAD_FILES"
            })
            continue

        group3_valid_edf_files.append(file_path)

print("\n" + "=" * 80)
print(f"Total valid EDF files across all Group 3 patients: {len(group3_valid_edf_files)}")

if len(group3_valid_edf_files) == 0:
    raise ValueError("No valid EDF files found across Group 3 patients.")

# Start from first valid file as reference
first_raw = load_raw_edf(group3_valid_edf_files[0])
first_raw = prepare_raw(first_raw)
reference_channels = first_raw.ch_names.copy()

group3_global_common = set(reference_channels)

# Find channels common across all valid Group 3 files
for file_path in group3_valid_edf_files[1:]:
    raw_tmp = load_raw_edf(file_path)
    raw_tmp = prepare_raw(raw_tmp)
    current_channels = set(raw_tmp.ch_names)
    group3_global_common = group3_global_common.intersection(current_channels)

# Keep order as close as possible to standard CHANNEL_ORDER
expected_channels = [ch for ch in CHANNEL_ORDER if ch in group3_global_common]

# Add any extra channels that are common but not in CHANNEL_ORDER
extra_common_channels = [
    ch for ch in reference_channels
    if ch in group3_global_common and ch not in expected_channels
]
expected_channels.extend(extra_common_channels)

print("\nGlobal common channels across all VALID Group 3 files:")
print(expected_channels)
print(f"Number of global common channels: {len(expected_channels)}")

# =========================================================
# LOOP THROUGH ALL GROUP 3 PATIENTS
# =========================================================

for patient_name in GROUP3_PATIENTS:
    print("\n" + "=" * 80)
    print(f"Processing patient: {patient_name}")

    patient_folder = RAW_ROOT / patient_name
    summary_path = patient_folder / f"{patient_name}-summary.txt"

    # output folder for this patient
    output_folder = PROCESSED_GROUP3 / patient_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # get all EDF files in this patient folder
    edf_files = sorted(patient_folder.glob("*.edf"))

    print(f"Found {len(edf_files)} EDF files")

    if len(edf_files) == 0:
        print(f"No EDF files found for {patient_name}, skipping...")
        continue

    # Read seizure information from summary file once
    summary_dict = parse_chbmit_summary(summary_path)

    print("\nUsing the same global Group 3 common channels for this patient:")
    print(expected_channels)
    print(f"Number of expected channels: {len(expected_channels)}")

    # summary rows for this patient only
    patient_summary_rows = []

    # known bad files for this patient
    patient_bad_files = set(BAD_FILES.get(patient_name, []))

    # =====================================================
    # LOOP THROUGH ALL FILES OF THIS PATIENT
    # =====================================================
    for file_path in edf_files:
        file_name = file_path.name.strip()
        seizure_intervals = summary_dict.get(file_name, [])

        # skip known bad files
        if file_name in patient_bad_files:
            group3_skipped_rows.append({
                "patient_name": patient_name,
                "file_name": file_name,
                "reason": "known bad file from BAD_FILES"
            })
            continue

        # Save path for this processed file
        save_path = output_folder / f"{Path(file_name).stem}_processed.npz"

        # If already processed, use existing file for summary and skip preprocessing
        if save_path.exists():
            try:
                saved_data = np.load(save_path)
                X = saved_data["X"]
                y = saved_data["y"]

                raw = load_raw_edf(file_path)
                raw = prepare_raw(raw)
                raw = reorder_channels(raw, expected_channels)

                bad_channels, channel_std = detect_bad_channels(raw, z_thresh=BAD_CHANNEL_Z_THRESH)

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
                group3_all_summary_rows.append(row)
                continue

            except Exception as e:
                print(f"Broken processed file found for {file_name}: {e}")
                print(f"Deleting broken file and reprocessing: {save_path}")

                try:
                    save_path.unlink()
                except Exception as delete_error:
                    print(f"Could not delete broken file: {delete_error}")

        print("\n" + "-" * 60)
        print(f"Processing file: {file_name}")
        print(f"Seizure intervals: {seizure_intervals}")

        try:
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

            print("EEG channels after reorder/common-channel selection:")
            print(raw.ch_names)
            print(f"Number of EEG channels after reorder: {len(raw.ch_names)}")

            # Check sampling rate
            print(f"Original sampling rate: {raw.info['sfreq']} Hz")

            # Apply filtering
            raw = apply_filters(
                raw,
                notch_freq=NOTCH_FREQ,
                l_freq=LOW_FREQ,
                h_freq=HIGH_FREQ
            )

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

            # Reduce file size before saving
            X = X.astype(np.float32)
            y = y.astype(np.int8)

            # Save processed result for this file
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
            group3_all_summary_rows.append(row)

        except Exception as e:
            print(f"Skipping {file_name} due to processing error: {e}")
            group3_skipped_rows.append({
                "patient_name": patient_name,
                "file_name": file_name,
                "reason": f"processing exception: {e}"
            })
            continue

    print(f"\nAll files in {patient_name} processed successfully.")

    # SAVE ONE SUMMARY FOR THE PATIENT
    df_patient_summary = pd.DataFrame(patient_summary_rows)

    patient_summary_save_path = output_folder / f"{patient_name}_processing_summary.csv"
    df_patient_summary.to_csv(patient_summary_save_path, index=False)

    print(f"\nPatient summary saved to: {patient_summary_save_path}")
    print(df_patient_summary.head())

# SAVE ONE COMBINED SUMMARY FOR ALL GROUP 3
df_group3_summary = pd.DataFrame(group3_all_summary_rows)

group3_summary_save_path = PROCESSED_GROUP3 / "group3_all_patients_summary.csv"
df_group3_summary.to_csv(group3_summary_save_path, index=False)

print("\n" + "=" * 80)
print("All Group 3 patients processed successfully.")
print(f"Combined Group 3 summary saved to: {group3_summary_save_path}")
print(df_group3_summary.head())

# SAVE SKIPPED FILES SUMMARY
df_group3_skipped = pd.DataFrame(group3_skipped_rows)
skipped_summary_save_path = PROCESSED_GROUP3 / "group3_skipped_files_summary.csv"
df_group3_skipped.to_csv(skipped_summary_save_path, index=False)

print(f"\nSkipped files summary saved to: {skipped_summary_save_path}")
print(df_group3_skipped.head())