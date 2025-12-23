import os
import nibabel as nib
import numpy as np
import csv

def analyze_and_save_to_csv(root_folder, output_csv="nifti_analysis_results.csv"):
    if not os.path.exists(root_folder):
        print(f"❌ Error: The folder '{root_folder}' does not exist.")
        return

    # List to store all rows of data
    results = []
    
    print(f"Scanning '{root_folder}' and subfolders...")
    print("-" * 60)

    # Walk through all folders
    for current_root, dirs, files in os.walk(root_folder):
        # Filter for .nii or .nii.gz files
        nii_files = sorted([f for f in files if f.endswith(('.nii', '.nii.gz'))])
        
        if nii_files:
            folder_name = os.path.basename(current_root)
            
            for f in nii_files:
                file_path = os.path.join(current_root, f)
                try:
                    # Load NIfTI
                    img = nib.load(file_path)
                    data = np.asanyarray(img.dataobj)
                    
                    # Calculate stats
                    d_min = np.min(data)
                    d_max = np.max(data)
                    d_shape = str(data.shape)
                    d_type = str(data.dtype)
                    unique_vals = np.unique(data)
                    
                    # Determine type
                    if len(unique_vals) < 20:
                        file_category = "MASK"
                        unique_str = str(unique_vals) # Save actual labels for masks
                    else:
                        file_category = "CT IMAGE"
                        unique_str = "Continuous"

                    # Add to results list
                    results.append({
                        "Folder": folder_name,
                        "File Name": f,
                        "Type": file_category,
                        "Shape": d_shape,
                        "Dtype": d_type,
                        "Min Value": d_min,
                        "Max Value": d_max,
                        "Unique Values": unique_str,
                        "Full Path": file_path
                    })
                    
                    print(f"Processed: {f} ({file_category})")
                    
                except Exception as e:
                    print(f"Error reading {f}: {e}")
                    results.append({
                        "Folder": folder_name,
                        "File Name": f,
                        "Type": "ERROR",
                        "Shape": "N/A",
                        "Dtype": "N/A",
                        "Min Value": "N/A",
                        "Max Value": "N/A",
                        "Unique Values": str(e),
                        "Full Path": file_path
                    })

    # --- Write to CSV ---
    if results:
        keys = ["Folder", "File Name", "Type", "Shape", "Dtype", "Min Value", "Max Value", "Unique Values", "Full Path"]
        
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            
            print("-" * 60)
            print(f"✅ Success! Observations saved to: {output_csv}")
            print(f"   Total files analyzed: {len(results)}")
            
        except IOError as e:
            print(f"❌ Error saving CSV file: {e}")
    else:
        print("No .nii files were found to analyze.")

# --- RUN THE FUNCTION ---
root_folder = 'data/covid_segmentation/Covid-19-2'  # Ensure this matches your path
analyze_and_save_to_csv(root_folder)