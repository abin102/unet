import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def preprocess_dataset(
    ct_dir, 
    mask_dir, 
    output_image_dir, 
    output_mask_dir
):
    """
    Slices 3D NIfTI volumes into 2D .npy files.
    Handles the nested folder structure of the CT images.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Get list of mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii')]

    print(f"Found {len(mask_files)} mask volumes. Starting conversion...")

    for mask_filename in tqdm(mask_files):
        # --- 1. Locate the Corresponding Image ---
        
        # The image folder usually shares the exact same name as the mask file
        # Example: Mask 'coronacases_001.nii' -> Folder 'coronacases_001.nii'
        expected_img_folder = os.path.join(ct_dir, mask_filename)
        
        real_image_path = None
        
        # Check if that folder exists
        if os.path.isdir(expected_img_folder):
            # Look inside the folder for the .nii file
            # Usually it's 'coronacases_org_001.nii' or similar
            files_inside = [f for f in os.listdir(expected_img_folder) if f.endswith('.nii')]
            
            if len(files_inside) > 0:
                # We found the image file inside the folder!
                real_image_path = os.path.join(expected_img_folder, files_inside[0])
            else:
                print(f"⚠️ Warning: Folder found for {mask_filename} but no .nii file inside.")
                continue
        else:
            # Fallback: Maybe it's not in a folder? (Check direct file)
            # This handles cases where the structure might be inconsistent
            direct_path = os.path.join(ct_dir, mask_filename)
            if os.path.exists(direct_path):
                 real_image_path = direct_path
            else:
                print(f"❌ Error: Could not find image for mask {mask_filename}")
                continue

        # --- 2. Load Volumes ---
        try:
            # Load Image (CT)
            img_nii = nib.load(real_image_path)
            img_vol = np.asanyarray(img_nii.dataobj) # Keep original dtype initially
            
            # Load Mask
            mask_path = os.path.join(mask_dir, mask_filename)
            mask_nii = nib.load(mask_path)
            mask_vol = np.asanyarray(mask_nii.dataobj)
            
        except Exception as e:
            print(f"Read Error on {mask_filename}: {e}")
            continue

        # --- 3. Safety Check: Orientations ---
        # Sometimes masks and images are rotated differently. 
        # For this dataset, we assume they align, but we check dimensions.
        if img_vol.shape != mask_vol.shape:
            # Sometimes the image is (512, 512, 300) and mask is (300, 512, 512)
            # If so, we might need to transpose. For now, we skip or warn.
            print(f"⚠️ Shape Mismatch! Img: {img_vol.shape} vs Mask: {mask_vol.shape}. Skipping.")
            continue

        # --- 4. Slice and Save ---
        # Iterate over the Z-axis (usually index 2)
        num_slices = img_vol.shape[2]
        
        for i in range(num_slices):
            img_slice = img_vol[:, :, i]
            mask_slice = mask_vol[:, :, i]

            # Filter: Only save if the slice has meaningful data? 
            # (Optional: uncomment next 2 lines to skip empty black images)
            # if np.max(mask_slice) == 0 and np.random.rand() > 0.1: 
            #    continue # Keep only 10% of empty slices to reduce dataset size

            # Create filename
            base_name = mask_filename.replace('.nii', '')
            save_name = f"{base_name}_slice_{i:03d}.npy"

            # Save
            # Image: Float32 (for training stability)
            np.save(os.path.join(output_image_dir, save_name), img_slice.astype(np.float32))
            
            # Mask: Uint8 (Classes 0, 1, 2)
            np.save(os.path.join(output_mask_dir, save_name), mask_slice.astype(np.uint8))

    print(f"\n✅ Processing Complete.")
    print(f"Images saved to: {output_image_dir}")

# --- CONFIGURATION (Based on your screenshots) ---
ct_source = 'data/covid_segmentation/Covid-19-2/Covid-19-2'
mask_source = 'data/covid_segmentation/Covid-19-2/InfectionMasks'

# Output folders
out_imgs = 'data/processed_data/images'
out_masks = 'data/processed_data/masks'

preprocess_dataset(ct_source, mask_source, out_imgs, out_masks)