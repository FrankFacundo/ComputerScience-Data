import os

from PIL import Image

# Define input and output directories
input_dir = "photos_webp"  # Change this to your directory
output_dir = "photos_png"  # Change this to your desired output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert all .webp files to .png
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith(".webp"):
        webp_path = os.path.join(input_dir, file_name)
        png_path = os.path.join(output_dir, file_name.replace(".webp", ".png"))

        with Image.open(webp_path) as img:
            img.save(png_path, "PNG")

print(f"Conversion completed! PNG files are saved in '{output_dir}'.")
