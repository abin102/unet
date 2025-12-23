import os
import shutil
import random
from tqdm import tqdm

def split_dataset_by_patient(data_root, val_ratio=0.2):
    # Paths
    images_dir = os.path.join(data_root, 'images')
    masks_dir = os.path.join(data_root, 'masks')
    
    # Create Train/Val folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(data_root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(data_root, split, 'masks'), exist_ok=True)

    # 1. Identify Unique Patients
    # We look at filenames like: "coronacases_001_slice_005.npy"
    # The Patient ID is everything before "_slice_"
    all_files = [f for f in os.listdir(images_dir) if f.endswith('.npy')]
    
    patient_ids = set()
    for f in all_files:
        # distinct_id = "coronacases_001"
        distinct_id = f.split('_slice_')[0] 
        patient_ids.add(distinct_id)
    
    patient_ids = list(patient_ids)
    random.shuffle(patient_ids)
    
    # 2. Split Patients
    num_val = int(len(patient_ids) * val_ratio)
    val_patients = patient_ids[:num_val]
    train_patients = patient_ids[num_val:]
    
    print(f"Total Patients: {len(patient_ids)}")
    print(f"Training on:    {len(train_patients)} patients")
    print(f"Validating on:  {len(val_patients)} patients")
    print("-" * 60)

    # 3. Move Files
    def move_files(patient_list, destination_split):
        print(f"Moving files to {destination_split}...")
        for pid in tqdm(patient_list):
            # Find all slices for this patient
            patient_slices = [f for f in all_files if f.startswith(pid + "_slice")]
            
            for f in patient_slices:
                # Move Image
                src_img = os.path.join(images_dir, f)
                dst_img = os.path.join(data_root, destination_split, 'images', f)
                shutil.move(src_img, dst_img)
                
                # Move Mask (if it exists)
                src_mask = os.path.join(masks_dir, f)
                dst_mask = os.path.join(data_root, destination_split, 'masks', f)
                if os.path.exists(src_mask):
                    shutil.move(src_mask, dst_mask)

    move_files(train_patients, 'train')
    move_files(val_patients, 'val')
    
    # Cleanup: Remove old empty folders
    try:
        os.rmdir(images_dir)
        os.rmdir(masks_dir)
    except:
        print("Note: Original folders not empty (maybe extra files), kept them.")

    print("\n✅ Splitting Complete.")
    print(f"Train Data: {data_root}/train")
    print(f"Val Data:   {data_root}/val")

# --- RUN IT ---
# Ensure this matches your folder name
processed_path = 'data/covid_segmentation/processed_data' 
split_dataset_by_patient(processed_path)