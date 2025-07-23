#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JpegLM + GPT-2 Image Captioning Training Script on Flickr8K
使用 JpegLM 作为视觉编码器，GPT-2 作为文本解码器
数据预处理参考 train_encoder.py，将图片转换为 JPEG 码流

示例运行命令:
python /root/autodl-tmp/MLLM/image_caption_train.py \
    --encoder_model /root/autodl-tmp/MLLM/models/jpeg-lm \
    --decoder_model gpt2 \
    --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm-gpt2-captioning \
    --image_size 224 \
    --max_enc_len 2048 \
    --max_dec_len 64 \
    --batch_size 1 \
    --epochs 3 \
    --learning_rate 5e-5 \
    --seed 42

或使用更大的解码器模型:
python flickr8k_image_caption_train.py \
    --encoder_model /path/to/jpeg-lm \
    --decoder_model gpt2-medium \
    --output_dir ./flickr8k_caption_gpt2medium \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 3e-5
"""

import argparse
import os
import sys
import random
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.modeling_utils import unwrap_model
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jpeg-lm', 'models'))
from jpeglm_encoder import create_seq2seq_model
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from data_utils import convert_img_to_bytes, create_preprocess_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Flickr8K Image Captioning Training")
    parser.add_argument("--encoder_model", type=str, required=True,
                        help="预训练 JpegLM 模型路径")
    parser.add_argument("--decoder_model", type=str, default="gpt2",
                        help="GPT-2 解码器模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./flickr8k_caption",
                        help="模型输出目录")
    parser.add_argument("--image_size", type=int, default=256,
                        help="输入图片预处理大小")
    parser.add_argument("--max_enc_len", type=int, default=1024,
                        help="编码器最大序列长度")
    parser.add_argument("--max_dec_len", type=int, default=64,
                        help="解码器最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def preprocess_dataset(examples, tokenizer, preprocess, max_enc_len, max_dec_len):
    # 处理图像
    img = preprocess(examples['image'])
    img = img.convert('RGB')
    jpeg_str = convert_img_to_bytes(img)
    
    # 编码器输入 - 更保守的长度限制
    # 为了避免位置编码问题，使用更小的序列长度
    safe_max_len = min(max_enc_len, 1024)  # 限制在1024以内
    
    # 在 tokenization 时就限制长度，避免警告
    jpeg_tokens = tokenizer(
        jpeg_str, 
        add_special_tokens=False,
        max_length=safe_max_len-1,  # 为 BOS token 留出空间
        truncation=True,
        return_tensors=None  # 确保返回列表而不是tensor
    )["input_ids"]
    
    # 确保不超过安全长度
    jpeg_tokens = jpeg_tokens[:safe_max_len-1]
    enc_ids = [tokenizer.bos_token_id] + jpeg_tokens
    
    # 再次确保总长度不超过限制
    enc_ids = enc_ids[:safe_max_len]
    
    # 填充到指定长度
    if len(enc_ids) < safe_max_len:
        enc_ids = enc_ids + [tokenizer.pad_token_id] * (safe_max_len - len(enc_ids))
    
    enc_mask = [1 if enc_ids[i] != tokenizer.pad_token_id else 0 for i in range(len(enc_ids))]

    # 解码器目标 - 修复数据访问逻辑
    if 'caption' in examples:
        caption = examples['caption']
    elif 'captions' in examples:
        captions_list = examples['captions']
        if isinstance(captions_list, list) and len(captions_list) > 0:
            caption = captions_list[0]
        else:
            caption = str(captions_list) if captions_list else "A photo."
    else:
        caption = "A photo."  # 默认标题
    
    # 确保 caption 是字符串
    if not isinstance(caption, str):
        caption = str(caption) if caption is not None else "A photo."
    
    dec = tokenizer(caption, max_length=max_dec_len, padding='max_length', truncation=True)
    # 将 pad_token 转换为 -100
    labels = [l if l != tokenizer.pad_token_id else -100 for l in dec['input_ids']]

    return { 'input_ids': enc_ids,
             'attention_mask': enc_mask,
             'labels': labels }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型和 tokenizer
    model = create_seq2seq_model(
        encoder_model_name_or_path=args.encoder_model,
        decoder_model_name_or_path=args.decoder_model,
        encoder_pooling_strategy='mean'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    model.to(device)

    # 加载 Flickr8K 数据集
    ds = load_dataset('jxie/flickr8k', split='train')
    # 划分训练和验证
    split = ds.train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split['train']
    val_ds = split['test']

    # 图像预处理
    preprocess = create_preprocess_transform(args.image_size)

    # Tokenize
    train_ds = train_ds.map(
        lambda ex: preprocess_dataset(ex, tokenizer, preprocess, args.max_enc_len, args.max_dec_len),
        remove_columns=train_ds.column_names, num_proc=12
    )
    val_ds = val_ds.map(
        lambda ex: preprocess_dataset(ex, tokenizer, preprocess, args.max_enc_len, args.max_dec_len),
        remove_columns=val_ds.column_names, num_proc=12,
    )

    # DataLoader
    def collate_fn(batch):
        enc_ids = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
        enc_mask = torch.tensor([b['attention_mask'] for b in batch], dtype=torch.long)
        labels = torch.tensor([b['labels'] for b in batch], dtype=torch.long)
        return { 'input_ids': enc_ids,
                 'attention_mask': enc_mask,
                 'labels': labels }
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_steps=500,
        eval_strategy='steps',
        eval_steps=500,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],  # 暂时禁用wandb
        remove_unused_columns=False,  # 保留数据集所有列
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
