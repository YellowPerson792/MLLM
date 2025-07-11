from datasets import load_dataset
import io
import json
import os
import binascii
import random

from PIL import Image
from os.path import basename

DATASET = "ylecun/mnist"
dataset = load_dataset(DATASET, split="train")
dataset = dataset.shuffle(seed=42)

def convert(example):
    buffer = io.BytesIO()
    example["image"].save(buffer, format="JPEG", quality=90, optimize=False, progressive=False)
    byte_data = buffer.getvalue()
    example["hex"] = binascii.hexlify(byte_data).decode("utf-8")
    return example

dataset = dataset.map(convert, remove_columns=["image"])

# 找到所有 hex 的最长公共前缀
def longest_common_prefix(strs):
    if not strs:
        return ''
    shortest = min(strs, key=len)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch:
                return shortest[:i]
    return shortest

# 先收集所有 hex 字符串，找到前缀
all_hex = [example["hex"] for example in dataset]
prefix = longest_common_prefix(all_hex)
print(f"hex公共前缀: {prefix}")

REMOVE_PREFIX = True  # 是否去除hex公共前缀
# 生成全量数据集
output_dir = f"/root/autodl-tmp/MLLM/datasets/{basename(DATASET)}"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{basename(DATASET)}_jpeg_factory2.jsonl")

with open(output_path, "w") as f:
    f.write('[\n')
    for idx, example in enumerate(dataset):
        hex_val = example["hex"][len(prefix):] if REMOVE_PREFIX else example["hex"]
        item = {
            "instruction": "请判断以下比特流图片的类别（0-9）",
            "input": hex_val,
            "output": str(example["label"]),
            "system": ""
        }
        json_str = json.dumps(item, ensure_ascii=False)
        if idx < len(dataset) - 1:
            f.write(json_str + ',\n')
        else:
            f.write(json_str + '\n')
    f.write(']\n')

# 生成小样本数据集
indices = random.sample(range(len(dataset)), 1000)
small_dataset = dataset.select(indices)
small_output_path = os.path.join(output_dir, f"{basename(DATASET)}_jpeg_small_factory2.jsonl")

with open(small_output_path, "w") as f:
    f.write('[\n')
    for idx, example in enumerate(small_dataset):
        hex_val = example["hex"][len(prefix):] if REMOVE_PREFIX else example["hex"]
        item = {
            "instruction": "请判断以下比特流图片的类别（0-9）",
            "input": hex_val,
            "output": str(example["label"]),
            "system": ""
        }
        json_str = json.dumps(item, ensure_ascii=False)
        if idx < len(small_dataset) - 1:
            f.write(json_str + ',\n')
        else:
            f.write(json_str + '\n')
    f.write(']\n')