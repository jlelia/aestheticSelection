"""
This script displays the most recent gallery image in a matplotlib figure.
The figure automatically refreshes when a new gallery image is generated.
"""

import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import glob
import re


def find_latest_gallery(output_dir="outputs"):
    """Find the most recent gallery image based on generation number."""
    gallery_files = glob.glob(os.path.join(output_dir, "gallery_gen_*.png"))

    if not gallery_files:
        return None

    # Extract generation numbers and find the highest one
    gen_numbers = []
    for file in gallery_files:
        match = re.search(r"gallery_gen_(\d+)\.png", file)
        if match:
            gen_numbers.append((int(match.group(1)), file))

    if not gen_numbers:
        return None

    # Return the path with the highest generation number
    return max(gen_numbers, key=lambda x: x[0])[1]


def display_gallery(output_dir="outputs", refresh_rate=2):
    """Display the latest gallery image and refresh periodically."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.ion()  # Turn on interactive mode

    current_gallery = None
    first_wait_log = True

    while True:
        latest_gallery = find_latest_gallery(output_dir)

        if latest_gallery and latest_gallery != current_gallery:
            current_gallery = latest_gallery
            # Clear the current figure
            plt.clf()
            # Load and display the image
            img = Image.open(current_gallery)
            plt.imshow(img)

            # Extract generation number for the title
            match = re.search(r"gallery_gen_(\d+)\.png", current_gallery)
            gen_num = match.group(1) if match else "?"

            plt.title(f"Gallery Generation {gen_num}")
            plt.axis("off")  # Hide axes
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)  # Small pause to update the figure

            print(f"Displaying gallery from generation {gen_num}")
            first_wait_log = False
        elif not latest_gallery and first_wait_log:
            print("Waiting for the first gallery image to appear in 'outputs'...")
            first_wait_log = False

        # Wait before checking for updates
        time.sleep(refresh_rate)


if __name__ == "__main__":
    # Use current directory for gallery images
    output_dir = "outputs"

    print(f"Gallery viewer started. Monitoring directory: {output_dir}")
    print("Press Ctrl+C to exit")

    try:
        display_gallery(output_dir)
    except KeyboardInterrupt:
        print("\nGallery viewer stopped.")
