import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import json
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import deque
import re

from config import (
    RAW_ROOT,
    PROCESSED_GROUP1,
    PROCESSED_GROUP2,
    PROCESSED_GROUP3,
    PROCESSED_GROUP4,
    GROUP1_PATIENTS,
    GROUP2_PATIENTS,
    GROUP3_PATIENTS,
    GROUP4_PATIENTS,
    BAD_FILES,
    CHANNEL_ORDER,
    FINAL_17_CHANNELS,
    FINAL17_EXCLUDED_PATIENTS,
    FINAL17_TRAIN_PATIENTS,
    FINAL17_VAL_PATIENTS,
    FINAL17_TEST_PATIENTS,
    WINDOW_SIZE_SEC,
    STEP_SEC,
    SOP_MIN,
    SPH_MIN
)


'''
some wrapping functions for processing files

processing just one file
processing all files of one patient
processing all files of one group
'''


def load_raw_edf(file_path):
    # This function loads the EDF file using MNE
    # preload=True means we load the data into memory so we can process it faster
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    return raw


def get_eeg_channels(raw):
    # in dataset EEG channels usually contain "-"
    # so we use that to filter only EEG channels and ignore others
    eeg_channels = [ch for ch in raw.ch_names if "-" in ch]
    return eeg_channels


def prepare_raw(raw):
    # First we keep only EEG channels
    eeg_channels = get_eeg_channels(raw)
    raw.pick(eeg_channels)

    # Then we explicitly tell MNE that these are EEG signals
    # this helps later when applying filters and processing
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    return raw


def apply_filters(raw, notch_freq=60, l_freq=0.5, h_freq=40):
    # We create a copy so we don’t modify the original raw signal directly
    raw_filtered = raw.copy()

    # Notch filter removes powerline noise (usually 50 or 60 Hz)
    raw_filtered.notch_filter(freqs=notch_freq, verbose=False)

    # Band-pass filter keeps only useful EEG frequencies
    # Here we keep signals between 0.5 Hz and 40 Hz
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    return raw_filtered


def zscore_normalize(data):
    # This function normalizes each channel separately
    # so that all channels have mean = 0 and std = 1

    data_norm = np.zeros_like(data)

    for i in range(data.shape[0]):
        mean = np.mean(data[i])
        std = np.std(data[i])

        # we add a very small number (1e-8) to avoid division by zero
        data_norm[i] = (data[i] - mean) / (std + 1e-8)

    return data_norm


def label_window(window_start_sec, window_end_sec, seizure_intervals):
    # We check if the current window overlaps with any seizure interval

    for seizure in seizure_intervals:
        seizure_start = seizure["start"]
        seizure_end = seizure["end"]

        # This condition checks overlap between window and seizure
        if window_start_sec < seizure_end and window_end_sec > seizure_start:
            return 1  # seizure window

    return 0  # non-seizure window




def parse_chbmit_summary(summary_path):
    seizure_info = {}

    current_file = None
    seizure_starts = []
    seizure_ends = []

    with open(summary_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith("File Name:"):
            if current_file is not None:
                seizure_info[current_file] = [
                    {"start": s, "end": e}
                    for s, e in zip(seizure_starts, seizure_ends)
                ]

            current_file = line.replace("File Name:", "").strip()
            seizure_starts = []
            seizure_ends = []

        elif "Seizure" in line and "Start Time" in line:
            match = re.search(r"(\d+)\s+seconds", line)
            if match:
                seizure_starts.append(int(match.group(1)))

        elif "Seizure" in line and "End Time" in line:
            match = re.search(r"(\d+)\s+seconds", line)
            if match:
                seizure_ends.append(int(match.group(1)))

    if current_file is not None:
        seizure_info[current_file] = [
            {"start": s, "end": e}
            for s, e in zip(seizure_starts, seizure_ends)
        ]

    return seizure_info


def detect_bad_channels(raw, z_thresh=2.0):
    # This function tries to find channels that behave very differently
    # from the others (too noisy or too flat)

    data = raw.get_data()
    channel_std = np.std(data, axis=1)

    mean_std = np.mean(channel_std)
    std_std = np.std(channel_std)

    bad_channels = []

    print("\nChannel STD values:")
    for ch, val in zip(raw.ch_names, channel_std):
        print(f"{ch}: {val:.6f}")

    for i, ch in enumerate(raw.ch_names):
        z_score = (channel_std[i] - mean_std) / (std_std + 1e-8)

        if abs(z_score) > z_thresh:
            bad_channels.append(ch)

    print("\nSuspicious channels:", bad_channels)
    return bad_channels, channel_std


def detect_spikes(raw, z_threshold=5):
    # This detects sudden sharp changes (spikes) in the signal

    data = raw.get_data()

    # Compute mean and std per channel
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)

    # Convert signal to z-score
    z_scores = (data - mean) / (std + 1e-8)

    # Mark values that are too large
    spike_mask = np.abs(z_scores) > z_threshold

    print("Spike count:", np.sum(spike_mask))
    return spike_mask


def detect_artifacts(raw, threshold=100e-6):
    # This detects very large amplitude values
    # which are usually artifacts (movement, muscle, etc.)

    data = raw.get_data()

    artifact_mask = np.abs(data) > threshold

    print("Artifact points:", np.sum(artifact_mask))
    return artifact_mask


def find_common_channels(file_list):
    # This is useful for Group 4 patients
    # It finds channels that exist in ALL files

    common_channels = None

    for file_path in file_list:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        eeg_channels = [ch for ch in raw.ch_names if "-" in ch]

        if common_channels is None:
            common_channels = set(eeg_channels)
        else:
            common_channels = common_channels.intersection(eeg_channels)

    return list(common_channels)


def create_windows_and_labels(data, sfreq, window_size_sec, step_sec, seizure_intervals):
    # This function splits the EEG signal into overlapping windows
    # and gives each window a label depending on seizure overlap

    window_size = int(window_size_sec * sfreq)
    step = int(step_sec * sfreq)

    X = []
    y = []

    for start in range(0, data.shape[1] - window_size + 1, step):
        end = start + window_size

        # Take one EEG window
        window = data[:, start:end]

        # Convert sample positions to time in seconds
        window_start_sec = start / sfreq
        window_end_sec = end / sfreq

        # Label the window using seizure intervals
        label = label_window(window_start_sec, window_end_sec, seizure_intervals)

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)


def save_processed_data(save_path, X, y):
    # Save windows and labels into a file
    # npz is good because it stores multiple arrays together

    np.savez_compressed(save_path, X=X, y=y)
    print(f"Saved: {save_path}")


def reorder_channels(raw, expected_channels):
    # This keeps channels in one fixed order across files
    raw = raw.copy()
    raw.pick(expected_channels)
    return raw


def find_common_channels_in_order(file_list):
    """
    Find channels that exist in all files, but keep the order
    of the first file so channel order stays consistent.
    """

    if len(file_list) == 0:
        return []

    # load first file and keep its EEG channel order as reference
    first_raw = load_raw_edf(file_list[0])
    first_raw = prepare_raw(first_raw)
    reference_channels = first_raw.ch_names.copy()

    # start with all reference channels as candidates
    common_channels = set(reference_channels)

    # check all remaining files
    for file_path in file_list[1:]:
        raw = load_raw_edf(file_path)
        raw = prepare_raw(raw)
        current_channels = set(raw.ch_names)
        common_channels = common_channels.intersection(current_channels)

    # keep only channels that are common, in the original reference order
    ordered_common_channels = [ch for ch in reference_channels if ch in common_channels]

    return ordered_common_channels


'''
Those are some visualization functions to make sure the processing is doing what we expect.
a customised functions to plot before/after filtering and the removed signal for a group of channels.
This is very useful to visually inspect the effect of filters and identify any issues.
also to check signal quality, frequency cleaning, segmentation, label correctness,
model inputs readiness and class usefulness.
'''


def plot_grouped_channel_comparison(
    raw_before,
    raw_after,
    start_sec=0,
    end_sec=10,
    channel_order=None,
    group_size=4,
    figsize=(18, 10)
):
    """
    Plot grouped EEG channels for comparison:
    - Column 1: Before filtering
    - Column 2: After filtering
    - Column 3: Removed signal (before - after)
    """

    # Default channel order
    if channel_order is None:
        channel_order = [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9"
        ]

    # Keep only channels that exist
    channel_order = [
        ch for ch in channel_order
        if ch in raw_before.ch_names and ch in raw_after.ch_names
    ]

    if len(channel_order) == 0:
        print("No matching channels found.")
        return

    # Time conversion
    sfreq = raw_before.info["sfreq"]
    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)

    time = raw_before.times[start_sample:end_sample]

    # Get data
    data_before = raw_before.get_data()
    data_after = raw_after.get_data()

    # Loop in groups
    for i in range(0, len(channel_order), group_size):
        group = channel_order[i:i + group_size]

        fig, axes = plt.subplots(len(group), 3, figsize=figsize, sharex=True)

        # Fix shape if only one channel
        if len(group) == 1:
            axes = np.expand_dims(axes, axis=0)

        for j, ch_name in enumerate(group):
            ch_idx = raw_before.ch_names.index(ch_name)

            signal_before = data_before[ch_idx, start_sample:end_sample]
            signal_after = data_after[ch_idx, start_sample:end_sample]
            signal_removed = signal_before - signal_after

            # Before
            axes[j, 0].plot(time, signal_before)
            axes[j, 0].set_title(f"{ch_name} - Before")
            axes[j, 0].grid(True)

            # After
            axes[j, 1].plot(time, signal_after)
            axes[j, 1].set_title(f"{ch_name} - After")
            axes[j, 1].grid(True)

            # Removed
            axes[j, 2].plot(time, signal_removed)
            axes[j, 2].set_title(f"{ch_name} - Removed")
            axes[j, 2].grid(True)

        # Labels
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        axes[-1, 2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()


def plot_raw_vs_normalized_signal(
    raw_data,
    normalized_data,
    sfreq,
    ch_names,
    start_sec=0,
    end_sec=10,
    channel_order=None,
    group_size=4,
    figsize=(16, 8)
):
    """
    Plot grouped comparison of raw vs normalized EEG signals.

    Column 1: Raw signal
    Column 2: Normalized signal
    """

    if channel_order is None:
        channel_order = ch_names.copy()

    channel_order = [ch for ch in channel_order if ch in ch_names]

    if len(channel_order) == 0:
        print("No matching channels found.")
        return

    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)
    time = np.arange(start_sample, end_sample) / sfreq

    for i in range(0, len(channel_order), group_size):
        group = channel_order[i:i + group_size]
        fig, axes = plt.subplots(len(group), 2, figsize=figsize, sharex=True)

        if len(group) == 1:
            axes = np.expand_dims(axes, axis=0)

        for j, ch_name in enumerate(group):
            ch_idx = ch_names.index(ch_name)

            signal_raw = raw_data[ch_idx, start_sample:end_sample]
            signal_norm = normalized_data[ch_idx, start_sample:end_sample]

            axes[j, 0].plot(time, signal_raw)
            axes[j, 0].set_title(f"{ch_name} - Raw")
            axes[j, 0].set_ylabel("Amplitude")
            axes[j, 0].grid(True)

            axes[j, 1].plot(time, signal_norm)
            axes[j, 1].set_title(f"{ch_name} - Normalized")
            axes[j, 1].set_ylabel("Z-score")
            axes[j, 1].grid(True)

        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()


def plot_suspicious_channels(
    data,
    sfreq,
    ch_names,
    suspicious_channels,
    start_sec=0,
    end_sec=10,
    figsize=(14, 8)
):
    """
    Plot suspicious channels for manual inspection.
    """

    if len(suspicious_channels) == 0:
        print("No suspicious channels to plot.")
        return

    suspicious_channels = [ch for ch in suspicious_channels if ch in ch_names]

    if len(suspicious_channels) == 0:
        print("None of the suspicious channels are present in ch_names.")
        return

    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)
    time = np.arange(start_sample, end_sample) / sfreq

    fig, axes = plt.subplots(len(suspicious_channels), 1, figsize=figsize, sharex=True)

    if len(suspicious_channels) == 1:
        axes = [axes]

    for ax, ch_name in zip(axes, suspicious_channels):
        ch_idx = ch_names.index(ch_name)
        signal = data[ch_idx, start_sample:end_sample]

        ax.plot(time, signal)
        ax.set_title(f"Suspicious Channel: {ch_name}")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def check_model_input_ready(X, y):
    """
    Check if final model input looks correct.
    """

    print("=== Model Input Check ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) != len(y):
        print("WARNING: Number of windows and labels do not match.")
    else:
        print("OK: Number of windows matches number of labels.")

    if X.ndim != 3:
        print("WARNING: X should usually be 3D -> (windows, channels, samples)")
    else:
        print("OK: X is 3D.")

    if y.ndim != 1:
        print("WARNING: y should usually be 1D.")
    else:
        print("OK: y is 1D.")

    unique_labels, counts = np.unique(y, return_counts=True)
    print("Unique labels:", unique_labels)
    print("Label counts:", dict(zip(unique_labels, counts)))

    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)

    print("X min:", np.min(X))
    print("X max:", np.max(X))
    print("X mean:", np.mean(X))
    print("X std:", np.std(X))

    if np.isnan(X).any():
        print("WARNING: X contains NaN values.")
    else:
        print("OK: No NaN values in X.")

    if np.isnan(y).any():
        print("WARNING: y contains NaN values.")
    else:
        print("OK: No NaN values in y.")


def plot_seizure_vs_nonseizure_windows(
    X,
    y,
    ch_names,
    channel_name=None,
    seizure_index=None,
    nonseizure_index=None,
    sfreq=256,
    figsize=(14, 8)
):
    """
    Plot one non-seizure window and one seizure window for comparison.
    """

    if channel_name is None:
        ch_idx = 0
        channel_name = ch_names[ch_idx]
    else:
        if channel_name not in ch_names:
            print(f"Channel {channel_name} not found.")
            return
        ch_idx = ch_names.index(channel_name)

    seizure_indices = np.where(y == 1)[0]
    nonseizure_indices = np.where(y == 0)[0]

    if len(nonseizure_indices) == 0:
        print("No non-seizure windows found.")
        return

    if nonseizure_index is None:
        nonseizure_index = nonseizure_indices[0]

    if len(seizure_indices) == 0:
        print("No seizure windows found in this data.")
        return

    if seizure_index is None:
        seizure_index = seizure_indices[0]

    signal_non = X[nonseizure_index, ch_idx, :]
    signal_seiz = X[seizure_index, ch_idx, :]

    time = np.arange(signal_non.shape[0]) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(time, signal_non)
    axes[0].set_title(f"Non-Seizure Window | Channel: {channel_name} | Label: {y[nonseizure_index]}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(time, signal_seiz)
    axes[1].set_title(f"Seizure Window | Channel: {channel_name} | Label: {y[seizure_index]}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_label_distribution(y, figsize=(6, 4)):
    """
    Plot distribution of labels.
    """

    unique_labels, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=figsize)
    plt.bar(unique_labels.astype(str), counts)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid(axis="y")
    plt.show()

    print("Label counts:", dict(zip(unique_labels, counts)))


def plot_psd_before_after(raw_before, raw_after, fmax=60):
    """
    Plot PSD before and after filtering.
    """

    print("PSD before filtering")
    raw_before.compute_psd(fmax=fmax).plot()

    print("PSD after filtering")
    raw_after.compute_psd(fmax=fmax).plot()


# =========================================================
# FUNCTIONS FOR FINAL 17 PREDICTION LOADING
# =========================================================

def get_all_processed_roots():
    return [
        PROCESSED_GROUP1,
        PROCESSED_GROUP2,
        PROCESSED_GROUP3,
        PROCESSED_GROUP4
    ]


def get_patient_prediction_npz_files(patient_name):
    """
    Search all processed group folders and collect all processed .npz files
    for one patient.
    """
    npz_files = []

    for root in get_all_processed_roots():
        patient_folder = root / patient_name

        if not patient_folder.exists():
            continue

        patient_npz_files = sorted(patient_folder.glob("*_processed.npz"))
        npz_files.extend(patient_npz_files)

    return npz_files


def get_split_prediction_npz_files(patient_list):
    all_files = []

    for patient_name in patient_list:
        patient_files = get_patient_prediction_npz_files(patient_name)
        all_files.extend(patient_files)

    return all_files


def get_final17_prediction_file_lists():
    train_files = get_split_prediction_npz_files(FINAL17_TRAIN_PATIENTS)
    val_files = get_split_prediction_npz_files(FINAL17_VAL_PATIENTS)
    test_files = get_split_prediction_npz_files(FINAL17_TEST_PATIENTS)

    if len(train_files) == 0:
        raise ValueError("No training prediction files found.")
    if len(val_files) == 0:
        raise ValueError("No validation prediction files found.")
    if len(test_files) == 0:
        raise ValueError("No test prediction files found.")

    return train_files, val_files, test_files


def get_valid_edf_files_for_patient(patient_name):
    patient_folder = RAW_ROOT / patient_name
    edf_files = sorted(patient_folder.glob("*.edf"))

    patient_bad_files = set(BAD_FILES.get(patient_name, []))
    valid_files = [f for f in edf_files if f.name not in patient_bad_files]

    return valid_files


def build_common_channel_order_from_files(file_list):
    """
    Find channels common to all files and keep them in a stable order.
    """

    if len(file_list) == 0:
        return []

    first_raw = load_raw_edf(file_list[0])
    first_raw = prepare_raw(first_raw)
    reference_channels = first_raw.ch_names.copy()

    common_set = set(reference_channels)

    for file_path in file_list[1:]:
        raw_tmp = load_raw_edf(file_path)
        raw_tmp = prepare_raw(raw_tmp)
        current_channels = set(raw_tmp.ch_names)
        common_set = common_set.intersection(current_channels)

    ordered_channels = [ch for ch in CHANNEL_ORDER if ch in common_set]

    extra_channels = [
        ch for ch in reference_channels
        if ch in common_set and ch not in ordered_channels
    ]
    ordered_channels.extend(extra_channels)

    return ordered_channels


def get_group_common_order(patient_list):
    all_valid_files = []

    for patient_name in patient_list:
        if patient_name in FINAL17_EXCLUDED_PATIENTS:
            continue

        valid_files = get_valid_edf_files_for_patient(patient_name)
        all_valid_files.extend(valid_files)

    return build_common_channel_order_from_files(all_valid_files)


def get_patient_specific_order(patient_name):
    valid_files = get_valid_edf_files_for_patient(patient_name)
    return build_common_channel_order_from_files(valid_files)


PATIENT_SOURCE_CHANNEL_MAP = None


def build_patient_source_channel_map():
    """
    Build the source channel order used for each patient's processed files.
    """
    patient_channel_map = {}

    group1_order = get_group_common_order(GROUP1_PATIENTS)
    for patient_name in GROUP1_PATIENTS:
        patient_channel_map[patient_name] = group1_order

    group2_order = get_group_common_order(GROUP2_PATIENTS)
    for patient_name in GROUP2_PATIENTS:
        patient_channel_map[patient_name] = group2_order

    group3_order = get_group_common_order(GROUP3_PATIENTS)
    for patient_name in GROUP3_PATIENTS:
        patient_channel_map[patient_name] = group3_order

    for patient_name in GROUP4_PATIENTS:
        if patient_name in FINAL17_EXCLUDED_PATIENTS:
            continue
        patient_channel_map[patient_name] = get_patient_specific_order(patient_name)

    return patient_channel_map


def get_patient_source_channel_map():
    global PATIENT_SOURCE_CHANNEL_MAP

    if PATIENT_SOURCE_CHANNEL_MAP is not None:
        return PATIENT_SOURCE_CHANNEL_MAP

    from config import CHANNEL_MAP_PATH

    # FAST PATH: load saved map if it already exists
    if CHANNEL_MAP_PATH.exists():
        with open(CHANNEL_MAP_PATH, "r") as f:
            PATIENT_SOURCE_CHANNEL_MAP = json.load(f)

        print(f"Loaded patient channel map from: {CHANNEL_MAP_PATH}")
        return PATIENT_SOURCE_CHANNEL_MAP

    # SLOW PATH: build it once from EDF files
    print("Patient channel map not found. Building it once from EDF files...")
    PATIENT_SOURCE_CHANNEL_MAP = build_patient_source_channel_map()

    CHANNEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CHANNEL_MAP_PATH, "w") as f:
        json.dump(PATIENT_SOURCE_CHANNEL_MAP, f, indent=4)

    print(f"Saved patient channel map to: {CHANNEL_MAP_PATH}")

    return PATIENT_SOURCE_CHANNEL_MAP


def select_final17_from_processed_X(X, source_channels):
    """
    Reorder/select the exact FINAL_17_CHANNELS from processed X.
    X shape: (windows, channels, samples)
    """

    missing = [ch for ch in FINAL_17_CHANNELS if ch not in source_channels]
    if len(missing) > 0:
        raise ValueError(f"Missing final 17 channels: {missing}")

    index_map = [source_channels.index(ch) for ch in FINAL_17_CHANNELS]
    X_final17 = X[:, index_map, :]

    return X_final17


def label_prediction_window(window_start_sec, window_end_sec, seizure_intervals, sop_min=10, sph_min=5):
    """
    Returns:
    1  -> preictal
    0  -> interictal
    -1 -> ignore
    """

    sop_sec = sop_min * 60
    sph_sec = sph_min * 60

    for seizure in seizure_intervals:
        seizure_start = seizure["start"]
        seizure_end = seizure["end"]

        preictal_start = seizure_start - sph_sec - sop_sec
        preictal_end = seizure_start - sph_sec

        # ignore ictal windows
        if window_start_sec < seizure_end and window_end_sec > seizure_start:
            return -1

        # ignore SPH gap windows
        if window_start_sec < seizure_start and window_end_sec > preictal_end:
            return -1

        # label preictal windows
        if window_start_sec < preictal_end and window_end_sec > preictal_start:
            return 1

    return 0


def relabel_file_for_prediction(n_windows, seizure_intervals, window_size_sec, step_sec, sop_min=30, sph_min=5):
    """
    Create prediction labels for all windows in one file.
    """

    y_pred = []

    for i in range(n_windows):
        window_start_sec = i * step_sec
        window_end_sec = window_start_sec + window_size_sec

        label = label_prediction_window(
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            seizure_intervals=seizure_intervals,
            sop_min=sop_min,
            sph_min=sph_min
        )
        y_pred.append(label)

    return np.array(y_pred, dtype=np.int8)


def load_prediction_npz_file(npz_path):
    """
    Load one processed .npz file, convert it to FINAL_17_CHANNELS,
    ignore saved detection labels, and rebuild prediction labels.
    """

    # load X only
    data = np.load(npz_path)
    X = data["X"]

    # get patient name from folder
    patient_name = npz_path.parent.name

    # get source channel order for this patient
    patient_channel_map = get_patient_source_channel_map()

    if patient_name not in patient_channel_map:
        raise ValueError(f"No source channel map found for patient {patient_name}")

    source_channels = patient_channel_map[patient_name]

    # select final 17 channels
    X = select_final17_from_processed_X(X, source_channels)

    # rebuild original EDF file name
    file_stem = npz_path.stem.replace("_processed", "")
    file_name = f"{file_stem}.edf"

    # load seizure intervals
    summary_path = RAW_ROOT / patient_name / f"{patient_name}-summary.txt"
    summary_dict = parse_chbmit_summary(summary_path)
    seizure_intervals = summary_dict.get(file_name, [])

    # rebuild prediction labels
    y_pred = relabel_file_for_prediction(
        n_windows=len(X),
        seizure_intervals=seizure_intervals,
        window_size_sec=WINDOW_SIZE_SEC,
        step_sec=STEP_SEC,
        sop_min=SOP_MIN,
        sph_min=SPH_MIN
    )

    # remove ignored windows
    keep_idx = np.where(y_pred != -1)[0]
    X = X[keep_idx]
    y_pred = y_pred[keep_idx]

    return X.astype(np.float32), y_pred.astype(np.int64)



def balance_prediction_data(X, y, train_ratio=4):
    """
    Keep all preictal windows and undersample interictal windows.
    """

    preictal_idx = np.where(y == 1)[0]
    interictal_idx = np.where(y == 0)[0]

    if len(preictal_idx) == 0:
        return X, y

    max_interictal = min(len(interictal_idx), len(preictal_idx) * train_ratio)

    selected_interictal = np.random.choice(
        interictal_idx,
        size=max_interictal,
        replace=False
    )

    selected_idx = np.concatenate([preictal_idx, selected_interictal])
    np.random.shuffle(selected_idx)

    X_balanced = np.asarray(X[selected_idx], dtype=np.float32)
    y_balanced = np.asarray(y[selected_idx], dtype=np.int64)

    return X_balanced, y_balanced


def prediction_batch_generator(
    file_paths,
    batch_size=32,
    balance_plan=None,
    preictal_fraction=0.25,
    shuffle=True,
    random_state=42
):
    """
    Hybrid generator:
    - uses dataset-level balancing through balance_plan
    - also forces each batch to contain preictal + interictal windows
    """

    rng = np.random.default_rng(random_state)

    n_preictal = max(1, int(round(batch_size * preictal_fraction)))
    n_interictal = batch_size - n_preictal

    if n_interictal <= 0:
        raise ValueError("preictal_fraction is too high for this batch size.")

    preictal_X_buffer = deque()
    preictal_y_buffer = deque()
    interictal_X_buffer = deque()
    interictal_y_buffer = deque()

    while True:
        working_files = file_paths.copy()

        if shuffle:
            random.shuffle(working_files)

        for npz_path in working_files:
            try:
                X, y = load_prediction_npz_file(npz_path)

                if len(X) == 0:
                    continue

                pre_idx = np.where(y == 1)[0]
                inter_idx = np.where(y == 0)[0]

                # Apply dataset-level balancing only to interictal windows
                if balance_plan is not None and len(inter_idx) > 0:
                    keep_mask = rng.random(len(inter_idx)) < balance_plan["keep_probability"]
                    inter_idx = inter_idx[keep_mask]

                if shuffle:
                    rng.shuffle(pre_idx)
                    rng.shuffle(inter_idx)

                # Add windows into separate buffers
                for idx in pre_idx:
                    preictal_X_buffer.append(X[idx])
                    preictal_y_buffer.append(y[idx])

                for idx in inter_idx:
                    interictal_X_buffer.append(X[idx])
                    interictal_y_buffer.append(y[idx])

                # Build balanced batches
                while (
                    len(preictal_X_buffer) >= n_preictal
                    and len(interictal_X_buffer) >= n_interictal
                ):
                    X_batch_list = []
                    y_batch_list = []

                    for _ in range(n_preictal):
                        X_batch_list.append(preictal_X_buffer.popleft())
                        y_batch_list.append(preictal_y_buffer.popleft())

                    for _ in range(n_interictal):
                        X_batch_list.append(interictal_X_buffer.popleft())
                        y_batch_list.append(interictal_y_buffer.popleft())

                    X_batch = np.stack(X_batch_list).astype(np.float32)
                    y_batch = np.array(y_batch_list, dtype=np.int64)

                    if shuffle:
                        order = rng.permutation(len(y_batch))
                        X_batch = X_batch[order]
                        y_batch = y_batch[order]

                    yield X_batch, y_batch

            except Exception as e:
                print(f"Skipping file in generator: {npz_path.name} -> {e}")
                continue



def load_small_prediction_split(file_paths):
    """
    Load validation or test split fully into RAM.
    Every file is converted to FINAL_17_CHANNELS first.
    """

    X_parts = []
    y_parts = []

    for npz_path in file_paths:
        try:
            X, y = load_prediction_npz_file(npz_path)

            if len(X) == 0:
                continue

            X_parts.append(np.asarray(X, dtype=np.float32))
            y_parts.append(np.asarray(y, dtype=np.int64))

        except Exception as e:
            print(f"Skipping file due to error: {npz_path.name} -> {e}")

    if len(X_parts) == 0:
        raise ValueError("No usable prediction windows found in this split.")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    return X_all, y_all

#this funcation for printing the trainingset 
#To make sure how is it balance to get the best results 


def prediction_eval_generator(file_paths, batch_size=32):
    """
    Generator for validation/test (NO balancing, NO shuffle)
    """

    while True:
        for npz_path in file_paths:
            try:
                X, y = load_prediction_npz_file(npz_path)

                if len(X) == 0:
                    continue

                for start in range(0, len(y), batch_size):
                    batch_idx = slice(start, start + batch_size)
                    yield X[batch_idx], y[batch_idx]

            except Exception as e:
                print(f"Skipping eval file: {npz_path.name} -> {e}")
                continue

def load_prediction_labels_only(npz_path):
    """
    Fast version for balance planning.
    It does NOT load full X data into RAM.
    It only reads the number of windows and rebuilds prediction labels.
    """

    data = np.load(npz_path, mmap_mode="r")
    n_windows = data["X"].shape[0]

    patient_name = npz_path.parent.name

    file_stem = npz_path.stem.replace("_processed", "")
    file_name = f"{file_stem}.edf"

    summary_path = RAW_ROOT / patient_name / f"{patient_name}-summary.txt"
    summary_dict = parse_chbmit_summary(summary_path)
    seizure_intervals = summary_dict.get(file_name, [])

    y_pred = relabel_file_for_prediction(
        n_windows=n_windows,
        seizure_intervals=seizure_intervals,
        window_size_sec=WINDOW_SIZE_SEC,
        step_sec=STEP_SEC,
        sop_min=SOP_MIN,
        sph_min=SPH_MIN
    )

    keep_idx = np.where(y_pred != -1)[0]
    y_pred = y_pred[keep_idx]

    return y_pred.astype(np.int64) 

def build_dataset_balance_plan(file_paths, train_ratio=4):
    """
    Build a low-memory dataset-level balancing plan.

    Fast version:
    - Does NOT load full EEG X data
    - Only rebuilds prediction labels and counts them
    """

    total_preictal = 0
    total_interictal = 0

    for npz_path in file_paths:
        try:
            y = load_prediction_labels_only(npz_path)

            if len(y) == 0:
                continue

            total_preictal += np.sum(y == 1)
            total_interictal += np.sum(y == 0)

        except Exception as e:
            print(f"Skipping file while building balance plan: {npz_path.name} -> {e}")

    if total_preictal == 0:
        keep_probability = 1.0
        target_interictal = total_interictal
    else:
        target_interictal = min(total_interictal, total_preictal * train_ratio)
        keep_probability = target_interictal / total_interictal if total_interictal > 0 else 1.0

    plan = {
        "total_preictal": int(total_preictal),
        "total_interictal": int(total_interictal),
        "target_interictal": int(target_interictal),
        "keep_probability": float(keep_probability),
        "train_ratio": train_ratio
    }

    return plan

#visulizationf for the trainign seizure set
def plot_preictal_windows(
    X,
    y,
    ch_names,
    num_samples=3,
    sfreq=256
):
    """
    Plot ONLY preictal windows (label=1)
    """

    preictal_idx = np.where(y == 1)[0]

    if len(preictal_idx) == 0:
        print(" No preictal windows found")
        return

    print(f" Found {len(preictal_idx)} preictal windows")

    selected = np.random.choice(preictal_idx, min(num_samples, len(preictal_idx)), replace=False)

    for i, idx in enumerate(selected):
        window = X[idx]

        plt.figure(figsize=(12, 5))
        for ch in range(window.shape[0]):
            plt.plot(window[ch] + ch * 5)  # offset for visibility

        plt.title(f"Preictal Window #{idx}")
        plt.xlabel("Time")
        plt.ylabel("Channels")
        plt.show()


def inspect_prediction_file(npz_path):
    """
    Print label summary for one single processed file only.
    """
    X, y = load_prediction_npz_file(npz_path)

    patient_name = npz_path.parent.name
    file_stem = npz_path.stem.replace("_processed", "")
    file_name = f"{file_stem}.edf"

    print(f"{patient_name}/{file_name}")
    print(f"Shape: {X.shape}")
    print(f"Preictal: {np.sum(y == 1)}")
    print(f"Interictal: {np.sum(y == 0)}")

#summary printet for the golbal plan
def print_dataset_balance_plan(plan):
    """
    Print only the final training totals for the low-memory dataset-level plan.
    """

    print("\n" + "=" * 70)
    print("FINAL TRAIN TOTALS")
    print("=" * 70)
    print(f"Before balancing -> preictal: {plan['total_preictal']}, interictal: {plan['total_interictal']}")
    print(f"Target after balancing -> preictal: {plan['total_preictal']}, interictal: {plan['target_interictal']}")

    if plan["total_preictal"] > 0:
        print(f"Planned balanced ratio (interictal:preictal) = {plan['target_interictal'] / plan['total_preictal']:.2f}:1")
    else:
        print("No preictal windows found in training set")

    print(f"Global interictal keep probability = {plan['keep_probability']:.6f}")

#funcation to convern eeg batched form to the needed one in keras cov1
def transpose_generator(generator):

    for X_batch, y_batch in generator:
        X_batch = np.transpose(X_batch, (0, 2, 1))
        yield X_batch, y_batch


#batches funcations 


from pathlib import Path
import numpy as np
import random

def cached_batch_generator(
    cache_dir,
    shuffle=True,
    use_sample_weights=False,
    pattern="*.npz"
):
    cache_dir = Path(cache_dir)

    batch_files = sorted(cache_dir.glob(pattern))

    if len(batch_files) == 0:
        raise ValueError(f"No cached batches found in {cache_dir} using pattern '{pattern}'")

    print("=" * 60)
    print("Cached batches found:", len(batch_files))
    print("Cache directory:", cache_dir)
    print("Pattern:", pattern)
    print("=" * 60)

    while True:
        if shuffle:
            random.shuffle(batch_files)

        for i, batch_file in enumerate(batch_files):
            data = np.load(batch_file)

            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.float32)

            if use_sample_weights:
                sample_weights = np.where(y == 1, 5.0, 1.0).astype(np.float32)
                yield X, y, sample_weights
            else:
                yield X, y

# frequency funcations
from scipy.signal import welch

def compute_frequency_features(window, sfreq=256):
    # window shape: (1280, 17)

    import numpy as np
    from scipy.signal import welch

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    features = []
    eps = 1e-8

    for ch in range(window.shape[1]):
        signal = window[:, ch]

        freqs, psd = welch(signal, fs=sfreq, nperseg=256)

        total_power = np.sum(psd) + eps

        band_powers = []

        for low, high in bands.values():
            mask = (freqs >= low) & (freqs <= high)
            power = np.sum(psd[mask])
            band_powers.append(power)

        band_powers = np.array(band_powers, dtype=np.float32)

        delta = band_powers[0]
        theta = band_powers[1]
        alpha = band_powers[2]
        beta = band_powers[3]

        # 1. Log band power: 5 features
        log_power = np.log(band_powers + eps)

        # 2. Relative band power: 5 features
        rel_power = band_powers / total_power

        # 3. Spectral entropy: 1 feature
        psd_norm = psd / (np.sum(psd) + eps)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + eps))
        spectral_entropy = spectral_entropy / np.log(len(psd_norm) + eps)

        # 4. Power ratios: 5 features
        power_ratios = np.array([
            delta / (alpha + eps),
            theta / (beta + eps),
            delta / (beta + eps),
            theta / (alpha + eps),
            (delta + theta) / (alpha + beta + eps)
        ], dtype=np.float32)

        # 5. Time-domain features: 3 features
        variance = np.var(signal)
        line_length = np.sum(np.abs(np.diff(signal)))
        energy = np.sum(signal ** 2)

        time_features = np.array([
            variance,
            line_length,
            energy
        ], dtype=np.float32)

        features.extend(log_power)
        features.extend(rel_power)
        features.append(spectral_entropy)
        features.extend(power_ratios)
        features.extend(time_features)

    return np.array(features, dtype=np.float32)