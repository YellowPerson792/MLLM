#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example: python /root/autodl-tmp/MLLM/jpeg-lm/train.py --model_name_or_path /root/autodl-tmp/MLLM/models/jpeg-lm --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm --seed 42 --lora_r 8 --lora_alpha 32 --logging_steps 5 --wandb_run_name jpeglm-mnist-v5 --batch_size 2 --gradient_accumulation_steps 8 --epochs 3 --learning_rate 2e-4 --train_subset_size 6000 --test_subset_size 1000 --fp16 --disable_wandb

import argparse
import torch
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
from torchvision import transforms

QUALITY = 25
CACHE_TABLES_FN = "cache_tables.jpg"
UNICODE_OFFSET = 10240
MAX_SEQ_LEN = 2048
preprocess = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(1.0,1.0), ratio=(1.0,1.0), antialias=True)
])

def convert_img_to_bytes(img: Image.Image) -> str:
    img.save(CACHE_TABLES_FN, format="JPEG", quality=QUALITY, subsampling="4:2:0", streamtype=1, restart_marker_blocks=1)
    buf = __import__('io').BytesIO()
    img.save(buf, format="JPEG", quality=QUALITY, subsampling="4:2:0", streamtype=2, restart_marker_blocks=1)
    data = buf.getvalue()
    return ''.join(chr(b + UNICODE_OFFSET) for b in data)

def tokenize_example(example, tokenizer):
    img = preprocess(example['image']) if preprocess else example['image'].resize((256,256))
    img = img.convert('RGB')
    tokenized = tokenizer(
        convert_img_to_bytes(img),
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        truncation=True,
    )
    return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'labels': example['label']}

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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=10, use_cache=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.gradient_checkpointing_enable()
    import torch.nn as nn
    target_modules = [name.split('.')[-1] for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        modules_to_save=[],
    )
    model = get_peft_model(model, peft_config).to(device)

    ds = load_dataset('ylecun/mnist')
    ds = ds.cast_column('image', ds['train'].features['image'])
    train_ds = ds['train']
    test_ds = ds['test']
    if args.train_subset_size:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.train_subset_size))
    if args.test_subset_size:
        test_ds = test_ds.shuffle(seed=args.seed).select(range(args.test_subset_size))
    train_ds = train_ds.map(lambda ex: tokenize_example(ex, tokenizer), batched=False, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(lambda ex: tokenize_example(ex, tokenizer), batched=False, remove_columns=test_ds.column_names)

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
    trainer.train()
    trainer.evaluate()
