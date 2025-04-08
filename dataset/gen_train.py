import json
import random

input_path = "path/to/medmcqa/train.json" #TODO
output_path = "path/to/medmcqa/train_5k.json" #TODO

SEED = 42  
random.seed(SEED)

with open(input_path, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

subset = random.sample(data, 5000)

with open(output_path, "w") as f:
    for entry in subset:
        f.write(json.dumps(entry) + "\n")

