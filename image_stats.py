import os
import numpy as np
from PIL import Image


def compute_mean_std_grayscale(image_dir):
    total_sum = 0.0
    total_sum_sq = 0.0
    total_pixels = 0

    # Walk through the directory structure
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        # Convert the image to grayscale ('L' mode)
                        img = img.convert('L')
                        # Convert image data to a numpy array and normalize to [0, 1]
                        img_np = np.array(img, dtype=np.float32) / 255.0
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

                # Update total pixel count and accumulators
                total_pixels += img_np.size
                total_sum += img_np.sum()
                total_sum_sq += (img_np ** 2).sum()

    # Compute mean and standard deviation
    if total_pixels != 0:
        mean = total_sum / total_pixels
        std = np.sqrt(total_sum_sq / total_pixels - mean ** 2)
    else:
        mean = 0
        std = 0
    return mean, std


if __name__ == '__main__':
    # Set the directory containing your images
    image_directory = '/Users/yizhang/train_data/'  # Change this to your image directory path
    mean, std = compute_mean_std_grayscale(image_directory)

    print("Grayscale Image Statistics:")
    print("Mean:", mean)
    print("Std:", std)
## Train: Mean=0.40; std=0.28