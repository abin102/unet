import numpy as np
import matplotlib.pyplot as plt
import os

# Path to one of your mask files
mask_path = "data/covid_segmentation/processed_data/train/masks/coronacases_002_slice_044.npy" # Change filename if needed

mask = np.load(mask_path)

print(f"Unique values: {np.unique(mask)}")

plt.imshow(mask, cmap='gray')
plt.title("White (1) should be Infection")
plt.show()