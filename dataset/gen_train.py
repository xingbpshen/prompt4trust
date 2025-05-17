import csv
import random

input_path = "/cim/data/PMC-VQA/train.csv"
output_path = "/cim/data/PMC-VQA/train_10.csv"

SEED = 42
random.seed(SEED)

# Read CSV into list of dicts
with open(input_path, "r", newline='') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Sample 5k entries (or fewer if the dataset is smaller)
sample_size = min(10, len(data))
sampled_data = random.sample(data, sample_size)

# Write sampled data back to CSV
with open(output_path, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(sampled_data)

