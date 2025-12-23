import os
import numpy as np
from tqdm import tqdm

def get_processed_stats(data_root):
    # Adjust these subfolder names if yours are different
    images_dir = os.path.join(data_root, 'images')
    masks_dir = os.path.join(data_root, 'masks')
    
    if not os.path.exists(masks_dir):
        print(f"❌ Error: Could not find folder: {masks_dir}")
        return

    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
    total_files = len(mask_files)
    
    print(f"Analyzing {total_files} slices in '{data_root}'...")
    print("-" * 60)

    # Counters
    positive_slices = 0  # Slices with infection
    negative_slices = 0  # Empty slices
    total_infection_pixels = 0
    
    # Iterate and check content
    for f in tqdm(mask_files):
        mask_path = os.path.join(masks_dir, f)
        
        try:
            # Load the mask
            mask = np.load(mask_path)
            
            # Check for infection (Assuming label 2 or 3 is infection, or >0 if binary)
            # If your previous script saved masks as: 0=Bg, 1=Lung, 2=Infection
            # Then we check for values >= 2. 
            # If it was binary (0=Bg, 1=Infection), check > 0.
            
            # calculating stats for "Infection" specifically
            # Adjust this threshold based on your specific labels!
            # If using the dataset from earlier: Infection is usually label 2 or 3.
            # If you are unsure, we count any pixel > 0 for now.
            infection_pixels = np.sum(mask > 0) 
            
            if infection_pixels > 0:
                positive_slices += 1
                total_infection_pixels += infection_pixels
            else:
                negative_slices += 1
                
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Stats Calculation
    if total_files == 0:
        print("No files found.")
        return

    pos_ratio = (positive_slices / total_files) * 100
    avg_area = total_infection_pixels / positive_slices if positive_slices > 0 else 0

    print("-" * 60)
    print(f"📊 DATASET STATISTICS")
    print("-" * 60)
    print(f"Total Slices:       {total_files}")
    print(f"Positive Slices:    {positive_slices} ({pos_ratio:.2f}%) -> Contains Infection")
    print(f"Negative Slices:    {negative_slices} ({100-pos_ratio:.2f}%) -> Empty/Background")
    print(f"Avg Infection Size: {avg_area:.0f} pixels per positive slice")
    print("-" * 60)
    
    if pos_ratio < 10:
        print("⚠️  WARNING: Severe Class Imbalance (<10% positive).")
        print("   Consider using Weighted CrossEntropy or Dice Loss.")
    else:
        print("✅  Balance looks acceptable.")

# --- RUN IT ---
processed_path = 'data/covid_segmentation/processed_data'
get_processed_stats(processed_path)