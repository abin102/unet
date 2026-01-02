import os
import numpy as np
from tqdm import tqdm
import sys

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
DATA_ROOT = "data/covid_segmentation/processed_data"
SPLITS = ["train", "val"]
# -------------------------------------------------------------

def check_dataset_integrity():
    print(f"🔍 Starting Data Scan in: {DATA_ROOT}\n")
    
    global_unique_values = set()
    files_with_artifacts = []
    total_files = 0
    
    for split in SPLITS:
        mask_dir = os.path.join(DATA_ROOT, split, "masks")
        
        if not os.path.exists(mask_dir):
            print(f"❌ Error: Directory not found: {mask_dir}")
            continue
            
        files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])
        total_files += len(files)
        
        print(f"📂 Checking '{split}' split ({len(files)} files)...")
        
        for fname in tqdm(files):
            fpath = os.path.join(mask_dir, fname)
            
            try:
                # Load mask
                mask = np.load(fpath)
                
                # Get unique values in this specific file
                unique_vals = np.unique(mask)
                
                # Update global tracker
                for v in unique_vals:
                    global_unique_values.add(v)
                
                # Check for "Bad" values
                # We expect either {0, 1} OR {0, 255}
                # Anything else (like 127, 2, 50) is likely a resize artifact
                is_clean_binary = np.array_equal(unique_vals, [0, 1]) or \
                                  np.array_equal(unique_vals, [0]) or \
                                  np.array_equal(unique_vals, [1])
                                  
                is_clean_raw = np.array_equal(unique_vals, [0, 255]) or \
                               np.array_equal(unique_vals, [255])

                if not (is_clean_binary or is_clean_raw):
                    files_with_artifacts.append((f"{split}/{fname}", unique_vals))

            except Exception as e:
                print(f"\n❌ CORRUPT FILE: {fpath} | Error: {e}")

    print("\n" + "="*50)
    print("📊 FINAL REPORT")
    print("="*50)
    print(f"Total Files Scanned: {total_files}")
    print(f"All Unique Values Found in Dataset: {sorted(list(global_unique_values))}")
    
    if len(files_with_artifacts) == 0:
        print("\n✅ STATUS: CLEAN. No weird artifacts found.")
        print("Your data is safe to use if you handle 0/255 conversion correctly.")
    else:
        print(f"\n⚠️  WARNING: Found {len(files_with_artifacts)} files with strange values!")
        print("These likely contain resize artifacts (e.g., 128, 64) that cause GPU crashes.")
        print("\nExamples of bad files:")
        for name, vals in files_with_artifacts[:10]:
            print(f"  - {name}: contains {vals}")
        if len(files_with_artifacts) > 10:
            print(f"  ... and {len(files_with_artifacts) - 10} more.")

if __name__ == "__main__":
    check_dataset_integrity()