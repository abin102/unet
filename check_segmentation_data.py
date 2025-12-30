import os
import numpy as np
from tqdm import tqdm

# Define your paths
base_dir = "data/covid_segmentation/processed_data"
splits = ["train", "val"]

print(f"Checking dataset at: {base_dir}")
print("Looking for values other than 0 and 1...\n")

found_issue = False

for split in splits:
    mask_dir = os.path.join(base_dir, split, "masks")
    
    if not os.path.exists(mask_dir):
        print(f"Skipping {split} (path not found: {mask_dir})")
        continue
        
    files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])
    
    print(f"Scanning {split} ({len(files)} files)...")
    
    for f in tqdm(files):
        path = os.path.join(mask_dir, f)
        try:
            mask = np.load(path)
            unique_vals = np.unique(mask)
            
            # Check if there are any values NOT in {0, 1}
            # We treat 255 as 'suspicious' for now to warn you, 
            # even though your config ignores it.
            invalid_vals = [x for x in unique_vals if x not in [0, 1]]
            
            if len(invalid_vals) > 0:
                print(f"\n[FAIL] File: {os.path.join(split, 'masks', f)}")
                print(f"       Unique values found: {unique_vals}")
                
                if 2 in unique_vals:
                    print("       -> CRITICAL: Found class '2'. This causes the CUDA error.")
                if 255 in unique_vals:
                    print("       -> INFO: Found '255' (ignore index).")
                
                found_issue = True
                # Remove 'break' if you want to see ALL bad files
                break 
                
        except Exception as e:
            print(f"\n[ERROR] Could not read {f}: {e}")

    if found_issue:
        break

if not found_issue:
    print("\n[SUCCESS] All masks contain only 0 and 1!")
else:
    print("\n[ACTION REQUIRED] Modify your __getitem__ to handle these values.")