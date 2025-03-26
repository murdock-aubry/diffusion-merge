import os
from PIL import Image

def convert_pngs_to_gif(input_directory, output_filename, duration=100, loop=0):
    """
    Convert a set of PNG files in a directory to a GIF.
    
    Parameters:
    - input_directory: Path to the directory containing PNG files
    - output_filename: Name of the output GIF file (include .gif extension)
    - duration: Time between frames in milliseconds (default 500ms)
    - loop: Number of times to loop the GIF (0 means infinite loop)
    
    Returns:
    - Path to the created GIF file
    """
    # Get all PNG files in the directory, sorted alphabetically
    png_files = sorted([
        os.path.join(input_directory, f) 
        for f in os.listdir(input_directory) 
        if f.lower().endswith('.png')
    ])
    
    # Check if there are any PNG files
    if not png_files:
        raise ValueError(f"No PNG files found in directory: {input_directory}")
    
    # Open the first image to use as a base
    images = []
    for file in png_files:
        img = Image.open(file)
        images.append(img)
    
    # Save the first image as the base for the GIF
    base_image = images[0]
    
    # Save the GIF
    base_image.save(
        output_filename, 
        save_all=True, 
        append_images=images[1:], 
        duration=duration, 
        loop=loop
    )
    
    # Close all images
    for img in images:
        img.close()
    
    return output_filename


convert_pngs_to_gif('gif', 'outputs/correlation.gif')