from datasets import load_dataset
import io
import json
import os
from PIL import Image
import binascii
import base64

dataset = load_dataset("uoft-cs/cifar10", split="train")
# dataset = dataset.filter(lambda example: example["label"] in [3, 6])

# 二分类标注
# def relabel(example):
#     example["label"] = 0 if example["label"] == 3 else 1
#     return example

# dataset = dataset.map(relabel)

def convert_to_bytes(example):
    buffer = io.BytesIO()
    example["img"].save(buffer, format="JPEG")
    byte_data = buffer.getvalue()
    example["byte_array"] = byte_data
    example["hex"] = binascii.hexlify(byte_data).decode("utf-8")
    example["base64"] = base64.b64encode(byte_data).decode("utf-8")
    return example

dataset = dataset.map(convert_to_bytes, remove_columns=["img"])

with open("/root/autodl-tmp/MLLM/datasets/cifar10_jpeg.jsonl", "w") as f:
    for example in dataset:
        f.write(json.dumps({
            "label": example["label"],
            "byte_array": list(example["byte_array"]),
            "hex": example['hex'],
            'base64': example['base64']
        }) + "\n")
example = dataset[40]

print(f"标签: {example['label']}")
print(f"图像字节流长度: {len(example['byte_array'])} 字节")

os.makedirs("/root/autodl-tmp/MLLM/datasets/cifar10_images", exist_ok=True)

img = Image.open(io.BytesIO(example["byte_array"]))  
img.save(f"/root/autodl-tmp/MLLM/datasets/cifar10_images/sample_0_label_{example['label']}.jpeg")


