import os
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import math

# 输出目录
out_dir = '/root/autodl-tmp/MLLM/datasets/cifar10/images'
os.makedirs(out_dir, exist_ok=True)

# 预处理方式与训练/推理一致
preprocess = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(1.0, 1.0), ratio=(1.0, 1.0), antialias=True)
])

# Tokenizer 路径（与训练/推理保持一致）
MODEL_PATH = '/root/autodl-tmp/MLLM/models/jpeg-lm'
QUALITY = 25
MAX_SEQ_LEN = 4096
FILTER_THRESHOLD = 3200  # 过滤阈值

def convert_img_to_bytes(img: Image.Image, quality: int = QUALITY) -> str:
    img.save("cache_tables.jpg", format="JPEG", quality=quality, subsampling="4:2:0", streamtype=1, restart_marker_blocks=1)
    with os.fdopen(os.dup(os.open("cache_tables.jpg", os.O_RDONLY)), 'rb') as _:
        pass  # just to ensure file is closed after save
    import io
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality, subsampling="4:2:0", streamtype=2, restart_marker_blocks=1)
        data = buf.getvalue()
    return ''.join(chr(b + 10240) for b in data)

def tokenize_example(example, tokenizer):
    img = preprocess(example["img"])
    img = img.convert("RGB")
    jpeg_str = convert_img_to_bytes(img, QUALITY)
    toks = tokenizer(jpeg_str, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True)
    # 获取实际token长度（无截断）
    actual_toks = tokenizer(jpeg_str, add_special_tokens=False)
    actual_len = len(actual_toks['input_ids'])
    return toks["input_ids"], img, actual_len

# 加载 cifar10 数据集
ds = load_dataset('uoft-cs/cifar10')
ds = ds.cast_column('img', ds['train'].features['img'])

# 选取前5000个样本
examples = ds['train'].select(range(5000))

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# 只保留：每个类别各保存一张示例图片和对应txt内容
save_txt_path = '/root/autodl-tmp/MLLM/datasets/cifar10/samples.txt'
os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
category_saved = set()
category_img_paths = {}
sample_txt_lines = []
for i, ex in enumerate(examples):
    label = ex['label']
    if label not in category_saved:
        input_ids, img_proc, actual_len = tokenize_example(ex, tokenizer)
        save_path = os.path.join(out_dir, f'category_{label}_example.png')
        img_proc.save(save_path)
        category_img_paths[label] = save_path
        category_saved.add(label)
        # 保存对应txt内容
        img = preprocess(ex["img"]).convert("RGB")
        jpeg_str = convert_img_to_bytes(img, QUALITY)
        jpeg_hex = ''.join(f'{ord(c)-10240:02x}' for c in jpeg_str)
        sample_txt_lines.append(f"Sample {i} label={label}\nJPEG hex: {jpeg_hex}\nToken ids: {input_ids}\n\n")
    if len(category_saved) == 10:
        break
if len(category_saved) < 10:
    raise ValueError(f"前5000样本仅包含类别: {sorted(category_saved)}，不足10类，请扩大样本范围！")
with open(save_txt_path, 'w', encoding='utf-8') as ftxt:
    ftxt.writelines(sample_txt_lines)
print('每个类别各保存一张示例图片:', category_img_paths)

# 2. 统计前2000个样本的平均token长度、最大token长度和标准差
lengths_2000 = []
filtered_2000_examples = []
for i, ex in enumerate(examples.select(range(2000))):
    input_ids, _, actual_len = tokenize_example(ex, tokenizer)
    if actual_len <= FILTER_THRESHOLD:
        lengths_2000.append(actual_len)
        filtered_2000_examples.append(ex)
if lengths_2000:
    avg_len = sum(lengths_2000)/len(lengths_2000)
    max_len = max(lengths_2000)
    std_len = math.sqrt(sum((l-avg_len)**2 for l in lengths_2000)/len(lengths_2000))
    print(f"前2000个样本中有效样本数 (实际长度≤{FILTER_THRESHOLD}): {len(lengths_2000)}")
    print(f"有效样本的平均实际 token 长度: {avg_len:.2f}")
    print(f"有效样本的最大实际 token 长度: {max_len}")
    print(f"有效样本的实际 token 长度标准差: {std_len:.2f}")
else:
    print("未统计到有效 token 长度")

# 3. 统计前2000个样本各类别比例（原始数据 vs 过滤后数据）
from collections import Counter
# 原始前2000个样本的类别比例
label_list_2000_original = [ex['label'] for ex in examples.select(range(2000))]
label_counter_2000_original = Counter(label_list_2000_original)
total_2000_original = len(label_list_2000_original)
print('原始前2000个样本各类别样本数:', dict(label_counter_2000_original))
print('原始前2000个样本各类别比例:', {k: v/total_2000_original for k, v in label_counter_2000_original.items()})

# 过滤后样本的类别比例
if filtered_2000_examples:
    label_list_2000_filtered = [ex['label'] for ex in filtered_2000_examples]
    label_counter_2000_filtered = Counter(label_list_2000_filtered)
    total_2000_filtered = len(label_list_2000_filtered)
    print('过滤后样本各类别样本数:', dict(label_counter_2000_filtered))
    print('过滤后样本各类别比例:', {k: v/total_2000_filtered for k, v in label_counter_2000_filtered.items()})
    
    # 比较差异
    print('\n各类别比例差异 (过滤后 - 原始):')
    original_ratios = {k: v/total_2000_original for k, v in label_counter_2000_original.items()}
    filtered_ratios = {k: v/total_2000_filtered for k, v in label_counter_2000_filtered.items()}
    for label in range(10):
        original_ratio = original_ratios.get(label, 0)
        filtered_ratio = filtered_ratios.get(label, 0)
        diff = filtered_ratio - original_ratio
        print(f"类别 {label}: {diff:+.4f} ({filtered_ratio:.4f} - {original_ratio:.4f})")
else:
    print("过滤后无有效样本")
