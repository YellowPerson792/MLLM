#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example: python /root/autodl-tmp/MLLM/jpeg-lm/train.py --model_name_or_path /root/autodl-fs/models/jpeg-lm --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm --seed 42 --lora_r 8 --lora_alpha 32 --logging_steps 5 --wandb_run_name jpeglm-mnist-v5 --batch_size 2 --gradient_accumulation_steps 8 --epochs 3 --learning_rate 2e-4 --train_subset_size 6000 --test_subset_size 1000 --fp16 --dataset_mode cifar10 --max_seq_len 1100 --disable_wandb

import argparse
import torch
import sys
import os
# 添加utils路径以导入统一的数据处理工具
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from data_utils import tokenize_example_for_training, get_dataset_config, create_preprocess_transform

from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType

MAX_SEQ_LEN = 2048

# 支持自动适配图片字段名和数据集名
def get_dataset_and_field(dataset_mode):
    if dataset_mode == 'mnist':
        return 'ylecun/mnist', 'image'
    elif dataset_mode == 'cifar10':
        return 'uoft-cs/cifar10', 'img'
    else:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--output_dir', default='./checkpoints')
    parser.add_argument('--train_subset_size', type=int)
    parser.add_argument('--test_subset_size', type=int)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--wandb_run_name', type=str)
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--dataset_mode', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN, help='Maximum token sequence length')
    args = parser.parse_args()

    set_seed(args.seed)
    # 最大token长度
    max_seq_len = args.max_seq_len
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=10, use_cache=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.gradient_checkpointing_enable()

    # model.config.pad_token_id = tokenizer.pad_token_id      # 适用于Qwen

    # 手动指定 LoRA 插入层
    LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj",  # 注意力
               "gate_proj", "up_proj", "down_proj"]   # MLP
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET,
    )
    model = get_peft_model(model, peft_config).to(device)

    # 自动适配数据集和图片字段
    dataset_name, image_field = get_dataset_and_field(args.dataset_mode)
    ds = load_dataset(dataset_name)
    # 兼容 mnist/cifar10
    if args.dataset_mode == 'mnist':
        ds = ds.cast_column('image', ds['train'].features['image'])
    elif args.dataset_mode == 'cifar10':
        ds = ds.cast_column('img', ds['train'].features['img'])
    train_ds = ds['train']
    test_ds = ds['test']
    if args.train_subset_size:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.train_subset_size))
    if args.test_subset_size:
        test_ds = test_ds.shuffle(seed=args.seed).select(range(args.test_subset_size))
    # 添加多进程处理加速数据预处理
    print("开始数据预处理...")
    import time
    start_time = time.time()

    # 创建预处理变换
    preprocess = create_preprocess_transform(96)  # 96x96 图像大小

    train_ds = train_ds.map(
        lambda ex: tokenize_example_for_training(ex, tokenizer, image_field, max_seq_len, preprocess), 
        batched=False, 
        remove_columns=train_ds.column_names,
        num_proc=12,  # 使用4个进程并行处理
        desc="处理训练数据"
    )
    test_ds = test_ds.map(
        lambda ex: tokenize_example_for_training(ex, tokenizer, image_field, max_seq_len, preprocess), 
        batched=False, 
        remove_columns=test_ds.column_names,
        num_proc=12,  # 使用4个进程并行处理
        desc="处理测试数据"
    )

    preprocess_time = time.time() - start_time
    print(f"数据预处理完成，耗时: {preprocess_time:.2f}秒")

    # ===== 样本token预览 =====
    preview_num = 1
    print("\n===== 样本token预览 =====")
    for idx in range(preview_num):
        sample = train_ds[idx]
        print(f"样本 {idx}:")
        print(f"  input_ids: {sample['input_ids']}")
        print(f"  input_ids解码: {tokenizer.decode([t for t in sample['input_ids'] if t != tokenizer.pad_token_id], skip_special_tokens=False)}")
        print(f"  labels: {sample['labels']}")
        print("------------------------")

    print("开始模型训练...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        eval_strategy='epoch',
        report_to=[] if args.disable_wandb else ['wandb'],
        run_name=None if args.disable_wandb else args.wandb_run_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )
    
    # 添加训练时间监控
    print("开始模型训练...")
    training_start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - training_start_time
    total_time = time.time() - start_time
    print(f"训练完成！")
    print(f"纯训练时间: {training_time:.2f}秒 ({training_time/60:.1f}分钟)")
    print(f"总时间(含预处理): {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"预处理占比: {(preprocess_time/total_time)*100:.1f}%")
