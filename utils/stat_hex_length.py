import json
import math
from datasets import load_dataset

DATA_FILE = "/root/autodl-tmp/MLLM/datasets/mnist/mnist_jpeg.jsonl"

# 载入数据
raw = load_dataset("json", data_files=DATA_FILE, split="train")

# 统计各类别的 hex 长度
class_lengths = {str(i): [] for i in range(10)}

for ex in raw:
    label = str(ex["label"]) if "label" in ex else str(ex["output"])  # 兼容不同字段
    hex_str = ex["hex"] if "hex" in ex else ex["input"]  # 兼容不同字段
    class_lengths[label].append(len(hex_str))

# 计算平均长度并输出
print("各类别 hex 平均长度及标准差：")
for i in range(10):
    label = str(i)
    lengths = class_lengths[label]
    avg = sum(lengths) / len(lengths) if lengths else 0
    std = math.sqrt(sum((l - avg) ** 2 for l in lengths) / len(lengths)) if lengths else 0
    print(f"  类别 {label}: 平均 {avg:.2f} 字符，标准差 {std:.2f}，共 {len(lengths)} 条")
