#!/usr/bin/env python
# -*- coding: utf-8 -*-
# JpegLM Encoder Architecture Training Script (Fixed)
# 多卡训练示例：
# torchrun --nproc_per_node=4 train_enc_cls.py --model_name_or_path /path/to/model --output_dir /path/to/output --batch_size 2 --gradient_accumulation_steps 8 --epochs 3 --learning_rate 2e-4 --classifier_lr 5e-4 --fp16 --dataset_mode mnist --pooling_strategy mean --max_seq_len 1024 --image_size 96
# 或使用 accelerate launch 方式
# accelerate launch --num_processes 2 /root/autodl-tmp/MLLM/train_enc_cls.py --model_name_or_path /root/autodl-tmp/MLLM/models/jpeg-lm --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm-encoder --seed 42 --lora_r 8 --lora_alpha 32 --logging_steps 5 --wandb_run_name jpeglm-encoder-mnist-v1 --batch_size 2 --gradient_accumulation_steps 8 --epochs 3 --learning_rate 2e-4 --classifier_lr 5e-4 --train_subset_size 6000 --test_subset_size 1000 --fp16 --dataset_mode mnist --pooling_strategy mean --max_seq_len 1024 --image_size 96 --disable_wandb
#
# 将 JpegLM 从生成式语言模型改造成 encoder 架构进行分类任务
# python /root/autodl-tmp/MLLM/train_enc_cls.py --model_name_or_path /root/autodl-tmp/MLLM/models/jpeg-lm --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm-encoder --seed 42 --lora_r 8 --lora_alpha 32 --logging_steps 5 --wandb_run_name jpeglm-encoder-mnist-v1 --batch_size 2 --gradient_accumulation_steps 8 --epochs 3 --learning_rate 2e-4 --classifier_lr 5e-4 --train_subset_size 6000 --test_subset_size 1000 --fp16 --dataset_mode mnist --pooling_strategy mean --max_seq_len 1024 --image_size 96 --disable_wandb

import argparse
import torch
import torch.nn as nn
import sys
import os
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# 添加 utils 路径以导入统一的数据处理工具
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from data_utils import tokenize_example_for_training, get_dataset_config, create_preprocess_transform

class JpegLMEncoderForClassification(PreTrainedModel):
    """
    将 JpegLM 改造成 Encoder 架构的分类模型
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # 加载预训练的 transformer 层（encoder 模式）
        self.model = AutoModel.from_pretrained(
            config.name_or_path,
            config=config
        )
        # 分类头
        self.dropout = nn.Dropout(0.3)
        self.pre_classifier = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 池化策略
        self.pooling_strategy = config.pooling_strategy
        # 执行父类初始化，然后自定义初始化分类头
        self.post_init()
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """自定义初始化分类头权重，确保初始 logits 接近 0"""
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        self.model.set_input_embeddings(embeddings)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )
        sequence_output = outputs.last_hidden_state
        # 池化
        if self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            pooled_output = torch.max(sequence_output, dim=1)[0]
        elif self.pooling_strategy == "cls":
            pooled_output = sequence_output[:, 0]
        else:  # last
            seq_lengths = attention_mask.sum(dim=1) - 1
            pooled_output = sequence_output[torch.arange(sequence_output.size(0)), seq_lengths]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return { 'loss': loss, 'logits': logits }


def get_dataset_and_field(dataset_mode):
    if dataset_mode == 'mnist':
        return 'ylecun/mnist', 'image'
    elif dataset_mode == 'cifar10':
        return 'uoft-cs/cifar10', 'img'
    else:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")


def create_encoder_model(model_name_or_path, num_labels=10, pooling_strategy='mean'):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = num_labels
    config.use_cache = False
    # 关键修正：切换为 encoder
    config.is_decoder = False
    config.add_cross_attention = False
    config.pooling_strategy = pooling_strategy
    config.name_or_path = model_name_or_path
    model = JpegLMEncoderForClassification(config)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JpegLM Encoder Architecture Training")
    parser.add_argument('--model_name_or_path', required=True, help="预训练 JpegLM 模型路径")
    parser.add_argument('--output_dir', default='./checkpoints_encoder', help="输出目录")
    parser.add_argument('--train_subset_size', type=int, default=6000, help="训练集子集大小")
    parser.add_argument('--test_subset_size', type=int, default=1000, help="测试集子集大小")
    parser.add_argument('--batch_size', type=int, default=2, help="批次大小")
    parser.add_argument('--epochs', type=int, default=3, help="训练轮数")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="梯度累积步数")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="LoRA 学习率 (默认 2e-4)")
    parser.add_argument('--classifier_lr', type=float, default=5e-4, help="分类头学习率 (默认 5e-4)")
    parser.add_argument('--save_steps', type=int, default=100, help="保存步数")
    parser.add_argument('--logging_steps', type=int, default=5, help="日志步数")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--fp16', action='store_true', help="使用 FP16")
    parser.add_argument('--bf16', action='store_true', help="使用 BF16")
    parser.add_argument('--wandb_run_name', type=str, default='jpeglm-encoder-mnist-v1', help="W&B 运行名称")
    parser.add_argument('--disable_wandb', action='store_true', help="禁用 W&B")
    parser.add_argument('--dataset_mode', type=str, default='mnist', choices=['mnist','cifar10'], help="数据集模式")
    parser.add_argument('--image_size', type=int, default=96, help="输入图像尺寸")
    parser.add_argument('--max_seq_len', type=int, default=2048, help="最大序列长度")
    parser.add_argument('--pooling_strategy', type=str, default='mean', choices=['mean','max','cls','last'], help="池化策略")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # 多卡与显存优化（简洁版）
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        print(f"[Distributed] local_rank={local_rank}, set device to cuda:{local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if not args.fp16 and not args.bf16:
            args.fp16 = True
            print("[Auto] 启用fp16")
    pin_memory = torch.cuda.is_available()
    num_workers = 12  # 与map的num_proc一致，或可参数化
    # 仅在多卡（nproc_per_node>1）时设置prefetch_factor，单卡不设置
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    prefetch = 2 if n_gpus > 1 and num_workers > 0 else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = create_encoder_model(
        args.model_name_or_path,
        num_labels=10,
        pooling_strategy=args.pooling_strategy
    )

    # 关键修复：在内部的 transformer 模型上启用梯度检查点，避免 OOM
    if hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("✓ 已启用梯度检查点以避免 OOM")

    # LoRA 设置
    LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj",
                   "gate_proj","up_proj","down_proj"]
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                              inference_mode=False,
                              r=args.lora_r,
                              lora_alpha=args.lora_alpha,
                              target_modules=LORA_TARGET,
                              lora_dropout=0.1,
                              modules_to_save=["classifier","pre_classifier"])
    model = get_peft_model(model, peft_config)
    model = model.to(device)

    # 加载数据集
    dataset_name, image_field = get_dataset_and_field(args.dataset_mode)
    ds = load_dataset(dataset_name)
    if args.dataset_mode == 'mnist':
        ds = ds.cast_column('image', ds['train'].features['image'])
    else:
        ds = ds.cast_column('img', ds['train'].features['img'])
    train_ds = ds['train']; test_ds = ds['test']
    if args.train_subset_size:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.train_subset_size))
    if args.test_subset_size:
        test_ds = test_ds.shuffle(seed=args.seed).select(range(args.test_subset_size))

    # 数据预处理：与 test.py 保持一致
    preprocess = create_preprocess_transform(args.image_size)
    
    train_ds = train_ds.map(
        lambda ex: tokenize_example_for_training(ex, tokenizer, image_field, args.max_seq_len, preprocess), 
        batched=False, 
        remove_columns=train_ds.column_names,
        num_proc=12,
        desc="处理训练数据"
    )
    test_ds = test_ds.map(
        lambda ex: tokenize_example_for_training(ex, tokenizer, image_field, args.max_seq_len, preprocess), 
        batched=False, 
        remove_columns=test_ds.column_names,
        num_proc=12,
        desc="处理测试数据"
    )
    
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

    # 自定义优化器
    def create_optimizer():
        lora_params, cls_params = [], []
        for n,p in model.named_parameters():
            if p.requires_grad:
                if 'classifier' in n or 'pre_classifier' in n:
                    cls_params.append(p)
                else:
                    lora_params.append(p)
        return torch.optim.AdamW([
            {'params': lora_params, 'lr': args.learning_rate, 'weight_decay': 0.01},
            {'params': cls_params, 'lr': args.classifier_lr, 'weight_decay': 0.01}
        ])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        eval_strategy='epoch',
        report_to=[] if args.disable_wandb else ['wandb'],
        run_name=None if args.disable_wandb else args.wandb_run_name,
        local_rank=local_rank if local_rank != -1 else None,
        dataloader_drop_last=True,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=num_workers,
        **({'dataloader_prefetch_factor': prefetch} if prefetch is not None else {}),
    )

    def compute_metrics(p):
        preds, label_ids = p.predictions, p.label_ids
        preds = preds.argmax(-1)
        return {'accuracy': (preds == label_ids).mean()}

    # 自定义Trainer，日志/上传wandb的损失除以梯度累计步数，但反向传播不变
    class LoggingAdjustedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            # 只影响日志和wandb，不影响反向传播
            if self.model.training and hasattr(self.args, 'gradient_accumulation_steps'):
                display_loss = loss / self.args.gradient_accumulation_steps
                if return_outputs:
                    return display_loss, outputs
                return display_loss
            else:
                if return_outputs:
                    return loss, outputs
                return loss

    trainer = LoggingAdjustedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(create_optimizer(), None)
    )

    # 开始训练
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)