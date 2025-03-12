import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load images from directory
image_dir = "outputs/unets/continuous1"
image_files = sorted(os.listdir(image_dir))  # Ensure images are sorted by iteration

# Convert images to grayscale arrays
images = [np.array(Image.open(os.path.join(image_dir, f)).convert("L")) for f in image_files]

# Compute pixel-wise difference between consecutive images
diff_magnitudes = []
for i in range(1, len(images)):
    diff = np.abs(images[i] - images[i-1])  # Absolute pixel difference
    diff_magnitudes.append(np.sum(diff))  # Sum of all pixel differences

# Normalize for better visualization
diff_magnitudes = np.array(diff_magnitudes) / np.max(diff_magnitudes)

# Plot the temporal derivative
plt.figure(figsize=(8, 4))
plt.plot(np.linspace(0, 1, len(diff_magnitudes)), diff_magnitudes, marker="o", linestyle="-", color = "#b97d4b")
plt.xlabel(r"Merged Factor $\lambda$")
plt.ylabel("Change Magnitude")
plt.title("Change in Images Across Merge")
plt.savefig("outputs/rate-of-change1.png", dpi = 400)
