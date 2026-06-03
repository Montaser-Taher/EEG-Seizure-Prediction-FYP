from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    FINAL_17_CHANNELS,
    WINDOW_SIZE_SEC,
    TARGET_SFREQ
)

from utils import (
    get_final17_prediction_file_lists,
    prediction_batch_generator,
    prediction_eval_generator,
    build_dataset_balance_plan,
    print_dataset_balance_plan,
    plot_preictal_windows
)


if __name__ == "__main__":


    print("LOADING FINAL 17 DATA FOR CNN MODEL")


    # 1. FILE LISTS
  
    train_files, val_files, test_files = get_final17_prediction_file_lists()

    print("\nDATA FILE SUMMARY")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

   
    # 2. BUILD LOW-MEMORY DATASET BALANCE PLAN
    
    print("\nBUILDING DATASET-LEVEL BALANCE PLAN...")
    balance_plan = build_dataset_balance_plan(train_files, train_ratio=4)
    print_dataset_balance_plan(balance_plan)

   
    # 3. TRAIN GENERATOR
    
    batch_size = 32

    train_generator = prediction_batch_generator(
    file_paths=train_files,
    batch_size=batch_size,
    balance_plan=balance_plan,
    preictal_fraction=0.25,
    shuffle=True,
    random_state=42
)

  
    # 4. VALIDATION + TEST GENERATORS
  
    val_generator = prediction_eval_generator(val_files, batch_size=batch_size)
    test_generator = prediction_eval_generator(test_files, batch_size=batch_size)

    # 5. SAMPLE TRAIN BATCH
 
    X_batch, y_batch = next(train_generator)

    print("\nSAMPLE TRAIN BATCH")

    print(f"Shape: {X_batch.shape}")
    print(f"Preictal: {np.sum(y_batch == 1)}")
    print(f"Interictal: {np.sum(y_batch == 0)}")

    
    # 6. CNN INPUT CHECK

    expected_timepoints = int(WINDOW_SIZE_SEC * TARGET_SFREQ)

    print("\data quaility of INPUT CHECK")
    print(f"Expected channels: {len(FINAL_17_CHANNELS)}")
    print(f"Actual channels: {X_batch.shape[1]}")
    print(f"Expected timepoints: {expected_timepoints}")
    print(f"Actual timepoints: {X_batch.shape[2]}")

    if X_batch.shape[1] == len(FINAL_17_CHANNELS):
        print("Channels OK")
    else:
        print("Channel mismatch")

    # 7. QUALITY CHECK

    print("\nQUALITY CHECK")
    print("-" * 60)
    print("NaN:", np.isnan(X_batch).any())
    print("Inf:", np.isinf(X_batch).any())
    print(f"Range: [{X_batch.min():.2f}, {X_batch.max():.2f}]")


    # 8. CLASS BALANCE IN SAMPLE BATCH
  
    print("\nCLASS BALANCE")
    pre = np.sum(y_batch == 1)
    inter = np.sum(y_batch == 0)

    if pre > 0:
        print(f"Ratio (interictal:preictal): {inter/pre:.2f}:1")
    else:
        print("No preictal in batch")

    # 9. VISUALIZE PREICTAL

    print("\nVISUALIZING PREICTAL WINDOWS...")
    plot_preictal_windows(X_batch, y_batch, FINAL_17_CHANNELS)

  
    print("READY FOR CNN TRAINING ")