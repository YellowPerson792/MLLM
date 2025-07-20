#!/usr/bin/env python
# -*- coding: utf-8 -*-
# JpegLM Encoder Architecture Training Script
# 将JpegLM从生成式语言模型改造成encoder架构进行分类任务
# python /root/autodl-tmp/MLLM/jpeg-lm/train_encoder.py --model_name_or_path /root/autodl-tmp/MLLM/models/jpeg-lm --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm-encoder --seed 42 --lora_r 8 --lora_alpha 32 --logging_steps 5 --wandb_run_name jpeglm-encoder-mnist-v1 --batch_size 2 --gradient_accumulation_steps 8 --epochs 3 --learning_rate 2e-4 --classifier_lr 5e-4 --train_subset_size 6000 --test_subset_size 1000 --fp16 --dataset_mode mnist --pooling_strategy mean --max_seq_len 1024 --disable_wandb

import argparse
import torch
import torch.nn as nn
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
    AutoModel,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

MAX_SEQ_LEN = 2048

class JpegLMEncoderForClassification(PreTrainedModel):
    """
    将JpegLM改造成Encoder架构的分类模型
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # 加载预训练的transformer层（不包含lm_head）
        self.model = AutoModel.from_pretrained(
            config.name_or_path, 
            config=config
        )
        
        # 移除原有的语言模型头，添加分类头
        self.dropout = nn.Dropout(0.3)  # 增加dropout防止过拟合
        
        # 添加层归一化稳定特征
        self.pre_classifier = nn.LayerNorm(config.hidden_size)
        
        # 简化为单层分类头，与train.py保持一致
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 池化策略
        self.pooling_strategy = "mean"  # 可选: "mean", "max", "cls", "last"
        
        # 自定义初始化分类头权重，确保初始损失正常
        self._init_classifier_weights()
        
        # 初始化分类头权重
        self.post_init()
    
    def _init_classifier_weights(self):
        """自定义初始化分类头权重，确保初始logits接近0"""
        # 使用标准的权重初始化
        nn.init.normal_(self.classifier.weight.data, mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias.data)
    
    def get_input_embeddings(self):
        """返回输入嵌入层，PEFT需要这个方法"""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, embeddings):
        """设置输入嵌入层"""
        self.model.set_input_embeddings(embeddings)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 获取transformer的输出 (encoder架构，支持双向注意力)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False  # encoder不需要缓存
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 应用不同的池化策略
        if self.pooling_strategy == "mean":
            # 平均池化（考虑attention_mask）
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = torch.mean(sequence_output, dim=1)
        elif self.pooling_strategy == "max":
            # 最大池化
            pooled_output = torch.max(sequence_output, dim=1)[0]
        elif self.pooling_strategy == "cls":
            # 使用第一个token (类似BERT的[CLS])
            pooled_output = sequence_output[:, 0]
        elif self.pooling_strategy == "last":
            # 使用最后一个有效token
            if attention_mask is not None:
                batch_size = sequence_output.size(0)
                seq_lengths = attention_mask.sum(dim=1) - 1  # 获取每个序列的实际长度
                pooled_output = sequence_output[torch.arange(batch_size), seq_lengths]
            else:
                pooled_output = sequence_output[:, -1]
        
        # 应用层归一化稳定特征
        pooled_output = self.pre_classifier(pooled_output)
        
        # 应用dropout和分类层
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }

# 支持自动适配图片字段名和数据集名
def get_dataset_and_field(dataset_mode):
    if dataset_mode == 'mnist':
        return 'ylecun/mnist', 'image'
    elif dataset_mode == 'cifar10':
        return 'uoft-cs/cifar10', 'img'
    else:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")

def create_encoder_model(model_name_or_path, num_labels=10):
    """
    创建encoder架构的JpegLM模型
    """
    # 加载配置
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = num_labels
    config.use_cache = False  # 训练时禁用缓存
    config.name_or_path = model_name_or_path
    
    # 添加分类相关配置
    if not hasattr(config, 'hidden_dropout_prob'):
        config.hidden_dropout_prob = 0.1
    
    # 创建encoder模型
    model = JpegLMEncoderForClassification(config)
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JpegLM Encoder Architecture Training")
    parser.add_argument('--model_name_or_path', required=True, help="预训练JpegLM模型路径")
    parser.add_argument('--output_dir', default='./checkpoints_encoder', help="输出目录")
    parser.add_argument('--train_subset_size', type=int, help="训练集子集大小")
    parser.add_argument('--test_subset_size', type=int, help="测试集子集大小")
    parser.add_argument('--batch_size', type=int, default=2, help="批次大小")
    parser.add_argument('--epochs', type=int, default=3, help="训练轮数")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="梯度累积步数")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="学习率")  # 与train.py保持一致
    parser.add_argument('--classifier_lr', type=float, default=5e-4, help="分类头学习率(通常比LoRA高)")
    parser.add_argument('--save_steps', type=int, default=100, help="保存步数")
    parser.add_argument('--logging_steps', type=int, default=10, help="日志步数")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--fp16', action='store_true', help="使用FP16")
    parser.add_argument('--bf16', action='store_true', help="使用BF16")
    parser.add_argument('--wandb_run_name', type=str, help="W&B运行名称")
    parser.add_argument('--disable_wandb', action='store_true', help="禁用W&B")
    parser.add_argument('--dataset_mode', type=str, default='cifar10', choices=['mnist', 'cifar10'], help="数据集模式")
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN, help="最大序列长度")
    parser.add_argument('--pooling_strategy', type=str, default='mean', choices=['mean', 'max', 'cls', 'last'], help="池化策略")
    args = parser.parse_args()

    set_seed(args.seed)
    max_seq_len = args.max_seq_len
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"=== JpegLM Encoder架构训练 ===")
    print(f"模型路径: {args.model_name_or_path}")
    print(f"数据集: {args.dataset_mode}")
    print(f"池化策略: {args.pooling_strategy}")
    print(f"设备: {device}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    
    # 创建encoder架构模型
    print("正在创建Encoder架构模型...")
    model = create_encoder_model(args.model_name_or_path, num_labels=10)
    model.pooling_strategy = args.pooling_strategy
    
    # 在内部的transformer模型上启用梯度检查点
    if hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("✓ 已启用梯度检查点")
    
    print(f"模型架构:")
    print(f"- Transformer层数: {model.config.num_hidden_layers}")
    print(f"- 隐藏维度: {model.config.hidden_size}")
    print(f"- 分类类别数: {model.config.num_labels}")
    print(f"- 池化策略: {model.pooling_strategy}")

    # 配置LoRA
    LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj",  # 注意力
                   "gate_proj", "up_proj", "down_proj"]   # MLP
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config).to(device)
    
    # 关键修复：确保分类头相关参数可以被训练
    print("\n=== 修复分类头训练问题 ===")
    classifier_modules = [
        'classifier',
        'pre_classifier'
    ]
    
    # 显式启用分类头参数的训练
    for name, param in model.named_parameters():
        if any(module in name for module in classifier_modules):
            param.requires_grad = True
            print(f"✓ 启用训练: {name} - 参数量: {param.numel():,}")
    
    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")
    
    # 详细显示可训练参数
    print(f"\n可训练参数详情:")
    lora_params = 0
    classifier_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora' in name.lower():
                lora_params += param.numel()
            elif any(module in name for module in classifier_modules):
                classifier_params += param.numel()
                print(f"  分类头: {name} - {param.numel():,}")
    
    print(f"LoRA参数: {lora_params:,}")
    print(f"分类头参数: {classifier_params:,}")
    
    if classifier_params == 0:
        print("⚠️  警告: 分类头参数未被训练，这会影响分类性能！")
    else:
        print("✅ 分类头参数已正确设置为可训练")

    # 加载数据集
    dataset_name, image_field = get_dataset_and_field(args.dataset_mode)
    print(f"正在加载数据集: {dataset_name}")
    ds = load_dataset(dataset_name)
    
    # 兼容不同数据集格式
    if args.dataset_mode == 'mnist':
        ds = ds.cast_column('image', ds['train'].features['image'])
    elif args.dataset_mode == 'cifar10':
        ds = ds.cast_column('img', ds['train'].features['img'])
    
    train_ds = ds['train']
    test_ds = ds['test']
    
    if args.train_subset_size:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.train_subset_size))
        print(f"使用训练集子集: {len(train_ds)}")
    if args.test_subset_size:
        test_ds = test_ds.shuffle(seed=args.seed).select(range(args.test_subset_size))
        print(f"使用测试集子集: {len(test_ds)}")

    # 数据预处理
    print("开始数据预处理...")
    import time
    start_time = time.time()
    
    preprocess = create_preprocess_transform(96)  # 96x96 图像大小
    
    train_ds = train_ds.map(
        lambda ex: tokenize_example_for_training(ex, tokenizer, image_field, max_seq_len, preprocess), 
        batched=False, 
        remove_columns=train_ds.column_names,
        num_proc=12,
        desc="处理训练数据"
    )
    test_ds = test_ds.map(
        lambda ex: tokenize_example_for_training(ex, tokenizer, image_field, max_seq_len, preprocess), 
        batched=False, 
        remove_columns=test_ds.column_names,
        num_proc=12,
        desc="处理测试数据"
    )
    
    preprocess_time = time.time() - start_time
    print(f"数据预处理完成，耗时: {preprocess_time:.2f}秒")

    # 自定义优化器：为分类头设置不同学习率
    def create_optimizer():
        """创建自定义优化器，为分类头设置更高的学习率"""
        classifier_params = []
        lora_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(module in name for module in ['classifier', 'pre_classifier']):
                    classifier_params.append(param)
                else:
                    lora_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                'params': lora_params,
                'lr': args.learning_rate,
                'weight_decay': 0.01,
            },
            {
                'params': classifier_params,
                'lr': args.classifier_lr,
                'weight_decay': 0.01,
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        print(f"✓ 创建自定义优化器:")
        print(f"  LoRA学习率: {args.learning_rate}")
        print(f"  分类头学习率: {args.classifier_lr}")
        print(f"  LoRA参数数量: {sum(p.numel() for p in lora_params):,}")
        print(f"  分类头参数数量: {sum(p.numel() for p in classifier_params):,}")
        
        return optimizer

    # 训练配置
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
    
    # 计算准确率的函数
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if predictions is None or labels is None:
            return {}
        predictions = predictions.argmax(axis=-1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(create_optimizer(), None),  # 使用自定义优化器
    )
    
    # 开始训练
    print("=== 开始Encoder架构训练 ===")
    training_start_time = time.time()
    
    trainer.train()
    
    # 训练完成统计
    training_time = time.time() - training_start_time
    total_time = time.time() - start_time
    
    print(f"\n=== 训练完成 ===")
    print(f"纯训练时间: {training_time:.2f}秒 ({training_time/60:.1f}分钟)")
    print(f"总时间(含预处理): {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"预处理占比: {(preprocess_time/total_time)*100:.1f}%")
    
    # 保存模型
    print(f"\n模型保存至: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("=== Encoder架构训练完成 ===")