# Example commands:
# python /root/autodl-tmp/MLLM/utils/check_prepocess.py --dataset cifar10 --image_size 96
# python /root/autodl-tmp/MLLM/utils/check_prepocess.py --dataset mnist --image_size 96
# python /root/autodl-tmp/MLLM/utils/check_prepocess.py --dataset imagenet100 --image_size 256
# python /root/autodl-tmp/MLLM/utils/check_prepocess.py --dataset cifar10 --filter --image_size 96
# python /root/autodl-tmp/MLLM/utils/check_prepocess.py --dataset cifar10 --bit_flip --bit_flip_prob 0.001 --image_size 96
# python /root/autodl-tmp/MLLM/utils/check_prepocess.py --dataset mnist --bit_flip --bit_flip_prob 0.005 --image_size 96

import os
import argparse
import glob
import random
import io
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import math
# 导入统一的数据处理工具
from data_utils import convert_img_to_bytes, create_preprocess_transform, get_dataset_config

# 支持自动适配图片字段名和数据集名
def get_dataset_config(dataset_mode):
    if dataset_mode == 'mnist':
        return 'ylecun/mnist', 'image', '/root/autodl-tmp/MLLM/datasets/mnist/images'
    elif dataset_mode == 'cifar10':
        return 'uoft-cs/cifar10', 'img', '/root/autodl-tmp/MLLM/datasets/cifar10/images'
    elif dataset_mode == 'imagenet100':
        return None, 'image', '/root/autodl-tmp/MLLM/datasets/ImageNet100/images'
    else:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")

def load_imagenet100_subset():
    """加载ImageNet100子数据集，随机选择10个类别"""
    imagenet_root = '/root/autodl-fs/datasets/imagenet100'
    all_classes = [d for d in os.listdir(imagenet_root) if os.path.isdir(os.path.join(imagenet_root, d))]
    
    if len(all_classes) < 10:
        raise ValueError(f"ImageNet100数据集类别数不足10个，实际只有{len(all_classes)}个")
    
    # 随机选择10个类别
    random.seed(42)
    selected_classes = sorted(random.sample(all_classes, 10))
    print(f"随机选择的10个类别: {selected_classes}")
    
    # 构建数据集，只取每个类别的前500张图片加速处理
    examples = []
    for label, class_name in enumerate(selected_classes):
        class_dir = os.path.join(imagenet_root, class_name)
        image_files = glob.glob(os.path.join(class_dir, '*.JPEG')) + glob.glob(os.path.join(class_dir, '*.jpg')) + glob.glob(os.path.join(class_dir, '*.png'))
        
        # 限制每个类别最多500张图片以加速处理
        image_files = image_files[:500]
        
        for img_path in image_files:
            examples.append({
                'image_path': img_path,
                'label': label,
                'class_name': class_name
            })
    
    # 随机打乱并返回
    random.shuffle(examples)
    print(f"ImageNet100子数据集总计 {len(examples)} 张图片（每类最多500张）")
    return examples

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'imagenet100'], help='Dataset to process')
parser.add_argument('--filter', action='store_true', help='Enable filtering by token length threshold')
parser.add_argument('--image_size', type=int, default=96, help='Image side length for preprocessing (e.g. 96 or 256)')
parser.add_argument('--bit_flip', action='store_true', help='Enable bit flip corruption for JPEG data')
parser.add_argument('--bit_flip_prob', type=float, default=0.001, help='Probability of flipping each bit (default: 0.001, i.e., 0.1%)')
args = parser.parse_args()

# 获取数据集配置
dataset_name, image_field, out_dir = get_dataset_config(args.dataset)
os.makedirs(out_dir, exist_ok=True)

# 预处理方式与训练/推理一致，使用统一的数据处理工具
preprocess = create_preprocess_transform(args.image_size)

# Tokenizer 路径（与训练/推理保持一致）
MODEL_PATH = '/root/autodl-tmp/MLLM/models/jpeg-lm'
QUALITY = 25
FILTER_THRESHOLD = 7800  # 过滤阈值（仅当启用过滤时使用）

def tokenize_example(example, tokenizer, image_field, save_reconstructed_image=False):
    if args.dataset == 'imagenet100':
        # ImageNet100直接从文件路径加载图片
        img = Image.open(example['image_path']).convert("RGB")
        img = preprocess(img)
    else:
        img = preprocess(example[image_field])
        img = img.convert("RGB")
    
    # 根据命令行参数决定是否启用比特反转
    bit_flip_prob = args.bit_flip_prob if args.bit_flip else 0.0
    jpeg_str = convert_img_to_bytes(img, QUALITY, bit_flip_prob=bit_flip_prob)
    
    # 只在需要保存图像时进行重构
    if save_reconstructed_image and bit_flip_prob > 0.0:
        # 将jpeg_str转换回字节数据
        byte_list = [ord(c) - 10240 for c in jpeg_str]  # UNICODE_OFFSET = 10240
        
        # 读取表格文件头（参考run.py的save_byte_image函数）
        cache_tables_path = "cache_tables.jpg"
        if os.path.exists(cache_tables_path):
            with open(cache_tables_path, 'rb') as f:
                hexdata = f.read().hex()
                table_int_list = [int(_e) for _e in bytearray.fromhex(hexdata)]
                table_int_list = table_int_list[2:-2]  # 移除前2字节(FF D8)和后2字节(FF D9)
            
            # 拼接完整的JPEG数据：前2字节 + 表格 + 后续数据
            complete_byte_list = byte_list[:2] + table_int_list + byte_list[2:]
            jpeg_bytes = bytes(complete_byte_list)
            
            # 尝试将字节数据转换回PIL图像
            try:
                with io.BytesIO(jpeg_bytes) as buf:
                    reconstructed_img = Image.open(buf).convert("RGB")
                    # 将重构的图像作为返回的图像
                    img = reconstructed_img
                    print(f"成功重构比特反转后的图像")
            except Exception as e:
                # 如果重构失败，直接返回损坏的字节数据用于强制保存
                print(f"图像重构失败，将强制保存损坏的JPEG数据: {str(e)[:50]}...")
                # 返回特殊标记和字节数据
                return [tokenizer.bos_token_id] + tokenizer(jpeg_str, add_special_tokens=False)["input_ids"], jpeg_bytes, len([tokenizer.bos_token_id] + tokenizer(jpeg_str, add_special_tokens=False)["input_ids"])
    
    # 不限制序列长度，直接tokenize，但添加bos_token_id保持与训练/推理一致
    input_ids = [tokenizer.bos_token_id] + tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
    # 获取实际token长度
    actual_len = len(input_ids)
    return input_ids, img, actual_len

# 加载数据集
if args.dataset == 'imagenet100':
    # ImageNet100特殊处理
    all_examples = load_imagenet100_subset()
    # 直接使用前3000个样本即可，无需5000个
    examples = all_examples[:3000] if len(all_examples) >= 3000 else all_examples
    print(f"ImageNet100实际使用样本数: {len(examples)}")
else:
    # HuggingFace数据集
    dataset_name, image_field, out_dir = get_dataset_config(args.dataset)
    ds = load_dataset(dataset_name)
    # 兼容不同数据集的图片字段名
    if args.dataset == 'mnist':
        ds = ds.cast_column('image', ds['train'].features['image'])
    elif args.dataset == 'cifar10':
        ds = ds.cast_column('img', ds['train'].features['img'])
    
    # 选取前5000个样本
    examples = ds['train'].select(range(5000))

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# 输出比特反转状态信息
if args.bit_flip:
    print(f"启用比特反转模式，反转概率: {args.bit_flip_prob:.6f} ({args.bit_flip_prob*100:.4f}%)")
else:
    print("未启用比特反转模式")

# 只保留：每个类别各随机保存一张示例图片和对应txt内容
import random
save_txt_path = os.path.join(os.path.dirname(out_dir), 'samples.txt')
os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
category_img_paths = {}
sample_txt_lines = []
# 按类别分组
from collections import defaultdict
label_to_indices = defaultdict(list)
for idx, ex in enumerate(examples):
    label_to_indices[ex['label']].append(idx)
# 随机选取每个类别一张
random.seed(42)
category_saved = set()
for label in range(10):
    indices = label_to_indices.get(label, [])
    if not indices:
        continue
    chosen_idx = random.choice(indices)
    ex = examples[chosen_idx]
    # 对于保存示例图片，启用图像重构
    input_ids, img_proc_or_bytes, actual_len = tokenize_example(ex, tokenizer, image_field, save_reconstructed_image=True)
    
    # 处理保存逻辑
    if isinstance(img_proc_or_bytes, bytes):
        # 如果返回的是字节数据（重构失败的情况），直接写入JPEG文件
        save_path = os.path.join(out_dir, f'category_{label}_example.jpg')
        with open(save_path, 'wb') as f:
            f.write(img_proc_or_bytes)
        print(f"类别 {label}: 强制保存损坏的JPEG数据到 {save_path}")
    else:
        # 如果返回的是PIL图像，保存为JPEG
        save_path = os.path.join(out_dir, f'category_{label}_example.jpg')
        img_proc_or_bytes.save(save_path, format='JPEG', quality=25)
        print(f"类别 {label}: 保存图像到 {save_path}")
    
    category_img_paths[label] = save_path
    category_saved.add(label)
    # 保存对应txt内容
    if args.dataset == 'imagenet100':
        img = Image.open(ex['image_path']).convert("RGB")
        img = preprocess(img)
    else:
        img = preprocess(ex[image_field]).convert("RGB")
    bit_flip_prob = args.bit_flip_prob if args.bit_flip else 0.0
    jpeg_str = convert_img_to_bytes(img, QUALITY, bit_flip_prob=bit_flip_prob)
    jpeg_hex = ''.join(f'{ord(c)-10240:02x}' for c in jpeg_str)
    sample_txt_lines.append(f"Sample {chosen_idx} label={label}\nJPEG hex: {jpeg_hex}\nToken ids: {input_ids}\n\n")
if len(category_saved) < 10:
    raise ValueError(f"前5000样本仅包含类别: {sorted(category_saved)}，不足10类，请扩大样本范围！")
with open(save_txt_path, 'w', encoding='utf-8') as ftxt:
    ftxt.writelines(sample_txt_lines)
print('每个类别各保存一张示例图片:', category_img_paths)

# 2. 统计前2000个样本的平均token长度、最大token长度和标准差
lengths_2000 = []
filtered_2000_examples = []
# 新增：按类别统计token长度
lengths_by_category = defaultdict(list)

# 为ImageNet100准备按比例采样的2000个样本
if args.dataset == 'imagenet100':
    # 按类别比例采样2000个样本
    samples_per_class = 200  # 每个类别200个样本
    sampled_examples = []
    for label in range(10):
        label_examples = [ex for ex in examples if ex['label'] == label]
        if len(label_examples) >= samples_per_class:
            sampled_examples.extend(random.sample(label_examples, samples_per_class))
        else:
            sampled_examples.extend(label_examples)
    examples_2000 = sampled_examples[:2000]

else:
    # 兼容HuggingFace Dataset和list两种情况
    if hasattr(examples, 'select'):
        # Dataset对象，切片和select都返回dict
        examples_2000 = examples.select(range(min(2000, len(examples))))
    else:
        # 普通list，需保证每个元素是dict
        examples_2000 = examples[:2000]
        # 如果元素不是dict，尝试转为dict（极端情况）
        if examples_2000 and not isinstance(examples_2000[0], dict):
            raise TypeError(f"examples_2000[0] 类型为{type(examples_2000[0])}，不是dict，请检查数据加载逻辑！")

for i, ex in enumerate(examples_2000):
    # 对于统计，不进行图像重构
    input_ids, _, actual_len = tokenize_example(ex, tokenizer, image_field, save_reconstructed_image=False)
    # 根据命令行参数决定是否过滤
    if not args.filter or actual_len <= FILTER_THRESHOLD:
        lengths_2000.append(actual_len)
        filtered_2000_examples.append(ex)
        # 新增：按类别记录token长度
        lengths_by_category[ex['label']].append(actual_len)
if lengths_2000:
    avg_len = sum(lengths_2000)/len(lengths_2000)
    max_len = max(lengths_2000)
    min_len = min(lengths_2000)
    std_len = math.sqrt(sum((l-avg_len)**2 for l in lengths_2000)/len(lengths_2000))
    if args.filter:
        print(f"前2000个样本中有效样本数 (实际长度≤{FILTER_THRESHOLD}): {len(lengths_2000)}")
    else:
        print(f"前2000个样本总数（无过滤）: {len(lengths_2000)}")
    print(f"样本的平均实际 token 长度: {avg_len:.2f}")
    print(f"样本的最大实际 token 长度: {max_len}")
    print(f"样本的最小实际 token 长度: {min_len}")
    print(f"样本的实际 token 长度标准差: {std_len:.2f}")
    
    # 新增：各类别平均token长度统计（仅对mnist和cifar10）
    if args.dataset in ['mnist', 'cifar10'] and lengths_by_category:
        print(f"\n各类别平均token长度统计:")
        for label in range(10):
            if label in lengths_by_category and lengths_by_category[label]:
                category_lengths = lengths_by_category[label]
                category_avg = sum(category_lengths) / len(category_lengths)
                category_min = min(category_lengths)
                category_max = max(category_lengths)
                category_std = math.sqrt(sum((l-category_avg)**2 for l in category_lengths)/len(category_lengths))
                print(f"  类别 {label}: 平均={category_avg:.2f}, 最小={category_min}, 最大={category_max}, 标准差={category_std:.2f}, 样本数={len(category_lengths)}")
            else:
                print(f"  类别 {label}: 无样本")
    
    # 统计各token长度区间的样本比例
    if args.dataset == 'imagenet100':
        step_size = 1000
    else:  # mnist和cifar10
        step_size = 200
    
    # 以平均长度为中心，向两侧延展统计
    center = int(avg_len)
    # 计算需要的区间数量，确保覆盖min到max的范围
    left_range = center - min_len
    right_range = max_len - center
    max_range = max(left_range, right_range)
    num_steps = (max_range // step_size) + 2  # 额外增加一些区间确保覆盖
    
    # 生成区间
    intervals = []
    for i in range(-num_steps, num_steps + 1):
        start = center + i * step_size
        end = center + (i + 1) * step_size
        intervals.append((start, end))
    
    # 统计每个区间的样本数
    interval_counts = {}
    for start, end in intervals:
        count = sum(1 for length in lengths_2000 if start <= length < end)
        if count > 0:  # 只显示有样本的区间
            interval_counts[(start, end)] = count
    
    # 按区间起始值排序并输出
    total_samples = len(lengths_2000)
    print(f"\nToken长度分布统计（步长={step_size}）:")
    for (start, end), count in sorted(interval_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"[{start}, {end}): {count}个样本 ({percentage:.2f}%)")
        
else:
    print("未统计到有效 token 长度")

# 3. 统计前2000个样本各类别比例（原始数据 vs 过滤后数据）
from collections import Counter
# 原始前2000个样本的类别比例
label_list_2000_original = [ex['label'] for ex in examples_2000]
label_counter_2000_original = Counter(label_list_2000_original)
total_2000_original = len(label_list_2000_original)
print('原始前2000个样本各类别样本数:', dict(label_counter_2000_original))
print('原始前2000个样本各类别比例:', {k: v/total_2000_original for k, v in label_counter_2000_original.items()})

# 过滤后样本的类别比例
if filtered_2000_examples:
    label_list_2000_filtered = [ex['label'] for ex in filtered_2000_examples]
    label_counter_2000_filtered = Counter(label_list_2000_filtered)
    total_2000_filtered = len(label_list_2000_filtered)
    if args.filter:
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
        print('样本各类别样本数（无过滤）:', dict(label_counter_2000_filtered))
        print('样本各类别比例（无过滤）:', {k: v/total_2000_filtered for k, v in label_counter_2000_filtered.items()})
else:
    print("未统计到有效样本")
