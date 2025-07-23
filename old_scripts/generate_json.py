from datasets import load_dataset
import io
import json
import os
from PIL import Image
import binascii
import base64
import random

DATASET = "ylecun/mnist"
# DATASET = "uoft-cs/cifar10"
# DATASET = "tanganke/stl10"
dataset = load_dataset(DATASET, split="train")
dataset = dataset.shuffle(seed=42)

def convert(example):
    buffer = io.BytesIO()
    example["image"].save(buffer, format="JPEG", quality=90, optimize=False, progressive=False)
    byte_data = buffer.getvalue()
    example["byte_array"] = byte_data
    example["hex"] = binascii.hexlify(byte_data).decode("utf-8")
    return example

dataset = dataset.map(convert, remove_columns=["image"])

with open(f"/root/autodl-tmp/MLLM/datasets/{os.path.basename(DATASET)}/{os.path.basename(DATASET)}_jpeg_factory.jsonl", "w") as f:
    for example in dataset:
        f.write(json.dumps({
            "label": example["label"],
            "byte_array": list(example["byte_array"]),
            "hex": example['hex'],
        }) + "\n")

indices = random.sample(range(len(dataset)), 1000)  # 从中随机选 100 个不重复的索引
small_dataset = dataset.select(indices)
with open(f"/root/autodl-tmp/MLLM/datasets/{os.path.basename(DATASET)}/{os.path.basename(DATASET)}_jpeg_small_factory.jsonl", "w") as f:
    for example in small_dataset:
        f.write(json.dumps({
            "label": example["label"],
            "byte_array": list(example["byte_array"]),
            "hex": example['hex'],
        }) + "\n")
        
length_sum = 0
for example in small_dataset:
    length_sum = length_sum + len(example["byte_array"])
    
average = length_sum / len(small_dataset)
print(f"Average byte array length: {average}")
    
for i in range(0,10):
    example = small_dataset.select([i])[0]
    img = Image.open(io.BytesIO(example["byte_array"]))  
    img.save(f"/root/autodl-tmp/MLLM/datasets/images/test_image_{i}.jpeg")


