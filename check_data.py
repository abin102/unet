import os
import numpy as np
from tqdm import tqdm

def analyze_masks(mask_dir):
    files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])
    print(f"Scanning {len(files)} mask files in: {mask_dir}")
    print("-" * 60)

    global_unique = set()
    max_val_found = 0
    bad_files = 0
    
    # We will check the first 500 files to save time, or remove [:500] to check all
    for f in tqdm(files[:1000]): 
        path = os.path.join(mask_dir, f)
        try:
            mask = np.load(path)
            unique_vals = np.unique(mask)
            
            # Update global stats
            for u in unique_vals:
                global_unique.add(u)
            
            curr_max = unique_vals.max()
            if curr_max > max_val_found:
                max_val_found = curr_max
            
            # Check for suspicious values (assuming you expect 0, 1, or 0, 1, 2)
            # If you think it is binary, anything > 1 is suspicious.
            # If you think it is 3-class, anything > 2 is suspicious.
            if curr_max > 1: 
                # Just printing the first few bad ones to keep log clean
                if bad_files < 5:
                    print(f"⚠️  Suspicious file: {f} | Values: {unique_vals}")
                bad_files += 1

        except Exception as e:
            print(f"❌ Error reading {f}: {e}")

    print("-" * 60)
    print(f"✅ REPORT")
    print(f"Unique values found across dataset: {sorted(list(global_unique))}")
    print(f"Maximum value found: {max_val_found}")
    print(f"Total files with values > 1: {bad_files}")
    
    if max_val_found > 2:
        print("\n🚨 CRITICAL ERROR FOUND:")
        print(f"Your dataset contains values like {max_val_found}.")
        print("If your config has 'n_classes: 2' or '3', this WILL crash CUDA.")
        print("SOLUTION: Use the 'CT_NpyDataset' fix I provided to clamp values.")
    elif max_val_found == 255:
        print("\n🚨 CRITICAL ERROR FOUND:")
        print("Your masks are 0-255 (Standard Image format).")
        print("You must divide by 255 or clamp to 1.")

# --- RUN CONFIG ---
# Update this path to where your .npy masks actually are
mask_path = 'data/covid_segmentation/processed_data/train/masks' 

if os.path.exists(mask_path):
    analyze_masks(mask_path)
else:
    # Fallback to checking val if train doesn't exist yet
    print("Train masks not found, checking Val...")
    analyze_masks('data/covid_segmentation/processed_data/val/masks')