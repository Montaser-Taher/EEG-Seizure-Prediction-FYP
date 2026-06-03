from pathlib import Path





# base project folder = EEG_FYP
BASE_DIR = Path(__file__).resolve().parent.parent

# Root folders
RAW_ROOT = BASE_DIR / "data" / "raw"      

PROCESSED_ROOT = Path("/mnt/d/EEG_FYP_DATA/processed")


# Filtering settings
NOTCH_FREQ = 60          
LOW_FREQ = 0.5           
HIGH_FREQ = 40          

# Sampling rate  256 HZ
TARGET_SFREQ = 256



# WINDOWING SETTINGS FOR the ML INPUT


WINDOW_SIZE_SEC = 5     
OVERLAP_SEC = 2.5        

# Step = how much we move each window
STEP_SEC = WINDOW_SIZE_SEC - OVERLAP_SEC



# PATIENTS GROUPS


# Group 1 they are clean and consistent
GROUP1_PATIENTS = [
    "chb01","chb02","chb03","chb04","chb05",
    "chb06","chb07","chb08","chb09","chb10",
    "chb23","chb24"
]

# Group 2 they are  consistent but different channels
GROUP2_PATIENTS = [
    "chb14","chb20","chb21","chb22"
]

# Group 3 are lightly messy (few bad files)
GROUP3_PATIENTS = [
    "chb16","chb17"
]

# Group 4 are so  messy (need flexible handling)
GROUP4_PATIENTS = [
    "chb11","chb12","chb13",
    "chb15","chb18","chb19"
]



# 5. BAD FILES (FOR GROUP 3 + 4)


BAD_FILES = {
    "chb16": ["chb16_16.edf", "chb16_17.edf"],
    "chb17": ["chb17c_12.edf"]
}



# 6. OUTPUT FOLDERS PER GROUP


PROCESSED_GROUP1 = PROCESSED_ROOT / "group1"
PROCESSED_GROUP2 = PROCESSED_ROOT / "group2"
PROCESSED_GROUP3 = PROCESSED_ROOT / "group3"
PROCESSED_GROUP4 = PROCESSED_ROOT / "group4"

FINAL_17_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]

# exclude inconsistent patient
FINAL17_EXCLUDED_PATIENTS = ["chb12"]

# patient-wise split 
FINAL17_TRAIN_PATIENTS = [
    "chb01", "chb02", "chb04", "chb06", "chb07", "chb08", "chb09", "chb10",
    "chb11", "chb15", "chb16", "chb18",
    "chb21", "chb22", "chb23", "chb13"

]

FINAL17_VAL_PATIENTS = [
    "chb03", "chb20"
]

FINAL17_TEST_PATIENTS = [
    "chb05", "chb14", "chb17", "chb19", "chb24"
]

# preictal window length before seizure
SOP_MIN = 10

# gap before seizure start
SPH_MIN = 5

#  DETECTION PARAMETERS FOR ANALYSIS ONLY and make dections 


# Bad channel detection
BAD_CHANNEL_Z_THRESH = 2.0

# Spike detection
SPIKE_Z_THRESH = 5

# Artifact detection 
ARTIFACT_THRESHOLD = 100e-6


CHANNEL_MAP_PATH = PROCESSED_ROOT / "patient_channel_map.json"


GROUP_SIZE = 4  # number of channels per plot group


#  CHANNEL ORDER FOR VISUALIZATION
# this order is based on the standard 10-20 system and can help with visual consistency to see how the channels are arranged on the scalp. It’s not strictly necessary for processing but can be nice for plotting and sanity checks.

CHANNEL_ORDER = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ","P7-T7","T7-FT9"
]