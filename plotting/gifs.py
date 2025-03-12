import os
import imageio.v2 as imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

def create_gif_from_pngs(directory, output_gif, duration=0.1):
    images = []

    png_files = sorted([f for f in os.listdir(directory) if f.endswith(".png")], reverse=True)

    for file in png_files:
        img_path = os.path.join(directory, file)
        images.append(imageio.imread(img_path))

    if images:
        imageio.mimsave(output_gif, images, duration=duration)
        print(f"GIF saved as {output_gif}")
    else:
        print("No PNG files found in the directory.")


# Load images from directory
image_dir = "outputs/unets/continuous2"
image_files = sorted(os.listdir(image_dir), reverse = True)  # Ensure images are sorted by iteration
images = [np.array(Image.open(os.path.join(image_dir, f)).convert("L")) for f in image_files]

# Get image dimensions
H, W = images[0].shape

# Create a colormap (e.g., viridis, plasma, or a custom blue-to-red scale)
cmap = plt.get_cmap("plasma")  # Change colormap here if desired

# Store frames for GIF
frames = []

# Compute pixel-wise difference between consecutive images
for i in range(1, len(images)):
    diff = np.abs(images[i] - images[i-1])  # Absolute pixel difference
    norm_diff = diff / 255.0  # Normalize between 0 and 1

    # Apply colormap
    color_diff = cmap(norm_diff)  # Apply color map to get RGBA values
    color_diff = (color_diff[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

    # Convert to PIL image and store frame
    frame = Image.fromarray(color_diff)
    frames.append(frame)

# Save as GIF
output_gif = "outputs/unets/change_visualization2.gif"
frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)

print(f"GIF saved at {output_gif}")


# Example usage:
# create_gif_from_pngs("outputs/unets/continuous2", "output2.gif", duration=5)