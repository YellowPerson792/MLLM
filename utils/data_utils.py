#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的数据预处理工具，用于train.py和evaluate_jpeglm.py
"""

import io
import os
import random
from PIL import Image
from torchvision import transforms

# 常量定义
QUALITY = 25
UNICODE_OFFSET = 10240

def convert_img_to_bytes(img: Image.Image, quality: int = QUALITY, bit_flip_prob: float = 0.0) -> str:
    """将PIL图像转换为JPEG字节串，使用Unicode编码
    
    Args:
        img: PIL图像对象
        quality: JPEG压缩质量
        bit_flip_prob: 比特反转概率，0.0表示不反转，0.001表示千分之一概率反转每个比特
    """
    # 保存JPEG表格文件（streamtype=1）
    img.save("cache_tables.jpg", format="JPEG", quality=quality, subsampling="4:2:0", streamtype=1, restart_marker_blocks=1)
    with os.fdopen(os.dup(os.open("cache_tables.jpg", os.O_RDONLY)), 'rb') as _:
        pass  # 确保文件保存后关闭
    
    # 获取JPEG数据（streamtype=2）
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality, subsampling="4:2:0", streamtype=2, restart_marker_blocks=1)
        data = buf.getvalue()
    
    # 如果启用比特反转，对JPEG数据进行随机比特翻转
    if bit_flip_prob > 0.0:
        data_list = list(data)
        for i in range(len(data_list)):
            byte_val = data_list[i]
            # 对每个字节的8个比特位进行概率性反转
            for bit_pos in range(8):
                if random.random() < bit_flip_prob:
                    # 反转第bit_pos位
                    byte_val ^= (1 << bit_pos)
            data_list[i] = byte_val
        data = bytes(data_list)
    
    # 转换为Unicode字符串
    return ''.join(chr(b + UNICODE_OFFSET) for b in data)

def create_preprocess_transform(image_size: int):
    """创建图像预处理变换"""
    return transforms.Compose([
        transforms.RandomResizedCrop((image_size, image_size), scale=(1.0, 1.0), ratio=(1.0, 1.0), antialias=True)
    ])

def tokenize_example_for_training(example, tokenizer, image_field, max_seq_len=2000, preprocess=None, bit_flip_prob=0.0):
    """
    训练时的样本tokenization函数
    添加bos_token_id，用于训练阶段
    """
    img = preprocess(example[image_field]) if preprocess else example[image_field].resize((256, 256))
    img = img.convert('RGB')
    jpeg_str = convert_img_to_bytes(img, QUALITY, bit_flip_prob=bit_flip_prob)
    
    # 加上bos_token_id，和run.py一致
    input_ids = [tokenizer.bos_token_id] + tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
    
    # 截断和pad
    input_ids = input_ids[:max_seq_len] + [tokenizer.pad_token_id] * max(0, max_seq_len - len(input_ids))
    attention_mask = [1 if i < len(input_ids) and input_ids[i] != tokenizer.pad_token_id else 0 for i in range(max_seq_len)]
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': example['label']}

def tokenize_example_for_evaluation(example, tokenizer, max_seq_len, image_key, preprocess=None, bit_flip_prob=0.0):
    """
    推理评估时的样本tokenization函数
    添加bos_token_id，用于评估阶段
    """
    img = preprocess(example[image_key]) if preprocess else example[image_key].resize((256, 256))
    img = img.convert("RGB")
    jpeg_str = convert_img_to_bytes(img, QUALITY, bit_flip_prob=bit_flip_prob)
    
    # 加上bos_token_id，和train.py一致
    input_ids = [tokenizer.bos_token_id] + tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
    
    # 截断和pad
    input_ids = input_ids[:max_seq_len] + [tokenizer.pad_token_id] * max(0, max_seq_len - len(input_ids))
    attention_mask = [1 if i < len(input_ids) and input_ids[i] != tokenizer.pad_token_id else 0 for i in range(max_seq_len)]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": example["label"]}

def get_dataset_config(dataset_mode):
    """获取数据集配置信息"""
    if dataset_mode == 'mnist':
        return 'ylecun/mnist', 'image', '/root/autodl-tmp/MLLM/datasets/mnist/images'
    elif dataset_mode == 'cifar10':
        return 'uoft-cs/cifar10', 'img', '/root/autodl-tmp/MLLM/datasets/cifar10/images'
    elif dataset_mode == 'imagenet100':
        return None, 'image', '/root/autodl-tmp/MLLM/datasets/ImageNet100/images'
    else:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")
