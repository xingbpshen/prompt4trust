import csv
import random
import os
from PIL import Image

input_path = "./data/PMC-VQA/train.csv"
output_path = "./data/PMC-VQA/train_5k.csv"
image_root = "./data/PMC-VQA/images"

SEED = 42
random.seed(SEED)
num_samples = 5000  

# Read CSV into list of dicts
with open(input_path, "r", newline='') as f:
    reader = csv.DictReader(f)
    data = list(reader)
    fieldnames = reader.fieldnames

used_indices = set()
valid_samples = []

while len(valid_samples) < num_samples and len(used_indices) < len(data):
    idx = random.randint(0, len(data) - 1)
    if idx in used_indices:
        continue
    used_indices.add(idx)

    row = data[idx]
    image_path = os.path.join(image_root, row["Figure_path"])

    if os.path.exists(image_path):
        try:
            # Try to open image to validate
            image = Image.open(image_path).convert("RGB")

            if image.height < 28 or image.width < 28:
                print(f"Skipping tiny image: {image_path} ({image.width}x{image.height})")
                continue

            valid_samples.append(row)
        except Exception as e:
            print(f"Skipping invalid or corrupted image: {image_path} ({e})")
    else:
        print(f"Skipping missing image path: {image_path}")

# Write valid samples to new CSV
with open(output_path, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(valid_samples)

print(f"Saved {len(valid_samples)} valid samples to {output_path}")



