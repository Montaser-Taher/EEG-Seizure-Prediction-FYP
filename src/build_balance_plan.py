from pathlib import Path
import sys
import json

PROJECT_ROOT = Path("/mnt/c/Users/MSI/Desktop/EEG_FYP")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import (
    get_final17_prediction_file_lists,
    build_dataset_balance_plan,
    print_dataset_balance_plan
)

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Load train/val/test files
train_files, val_files, test_files = get_final17_prediction_file_lists()

print("Train files:", len(train_files))
print("Validation files:", len(val_files))
print("Test files:", len(test_files))

# Rebuild balance plan using fixed parser/labels
balance_plan = build_dataset_balance_plan(
    file_paths=train_files,
    train_ratio=4
)

print_dataset_balance_plan(balance_plan)

# Save new balance plan
save_path = results_dir / "balance_plan_train_ratio_4.json"

with open(save_path, "w") as f:
    json.dump(balance_plan, f, indent=4)

print("Saved new balance plan to:", save_path)