# 示例运行命令：
# python /root/autodl-tmp/MLLM/jpeg-lm/classification_encoder_train.py --model_name_or_path /root/autodl-fs/models/jpeg-lm --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm-encoder --seed 42 --lora_r 8 --lora_alpha 32 --logging_steps 5 --wandb_run_name jpeglm-encoder-mnist-v1 --batch_size 2 --gradient_accumulation_steps 8 --epochs 3 --learning_rate 2e-4 --classifier_lr 5e-4 --train_subset_size 6000 --test_subset_size 1000 --fp16 --pooling_strategy mean --max_seq_len 1024 --image_size 96 --dataset_mode mnist --disable_wandb --probe_layers "['classifier','pre_classifier']"

# 将 JpegLM 从生成式语言模型改造成 encoder 架构进行分类任务
# python /root/autodl-tmp/MLLM/jpeg-lm/classification_encoder_train.py --model_name_or_path /root/autodl-fs/models/jpeg-lm --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm-encoder --seed 42 --lora_r 8 --lora_alpha 32 --logging_steps 5 --wandb_run_name jpeglm-encoder-mnist-v1 --batch_size 2 --gradient_accumulation_steps 8 --epochs 3 --learning_rate 2e-4 --classifier_lr 5e-4 --train_subset_size 6000 --test_subset_size 1000 --fp16 --pooling_strategy mean --max_seq_len 1024 --image_size 96 --dataset_mode mnist --disable_wandb

import argparse
import torch
import torch.nn as nn
import sys
import os
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# ===== 冻结和解冻指定层 =====
def freeze_and_unfreeze(model, probe_layers=None):
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 解冻分类头和自定义层
    probe_layers = probe_layers or ['classifier', 'pre_classifier']
    for name, param in model.named_parameters():
        for probe in probe_layers:
            if probe in name:
                param.requires_grad = True

# 添加 utils 路径以导入统一的数据处理工具
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from data_utils import tokenize_example_for_training, get_dataset_config, create_preprocess_transform

# 导入自定义的模型结构
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from jpeglm_encoder import create_jpeglm_encoder_model


def get_dataset_and_field(dataset_mode):
    if dataset_mode == 'mnist':
        return 'ylecun/mnist', 'image'
    elif dataset_mode == 'cifar10':
        return 'uoft-cs/cifar10', 'img'
    elif dataset_mode == 'imagenet100':
        return None, None  # 特殊处理，见下方
    else:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")


def create_encoder_model(model_name_or_path, num_labels=10, pooling_strategy='mean'):
    """使用外部模型定义创建编码器模型"""
    return create_jpeglm_encoder_model(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        pooling_strategy=pooling_strategy,
        classifier_dropout=0.1
    )

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
    parser.add_argument('--dataset_mode', type=str, default='mnist', choices=['mnist','cifar10','imagenet100'], help="数据集模式")
    parser.add_argument('--imagenet100_dir', type=str, default='/root/autodl-fs/datasets/imagenet100', help="imagenet100根目录")
    parser.add_argument('--image_size', type=int, default=96, help="输入图像尺寸")
    parser.add_argument('--max_seq_len', type=int, default=2048, help="最大序列长度")
    parser.add_argument('--pooling_strategy', type=str, default='mean', choices=['mean','max','cls','last'], help="池化策略")
    parser.add_argument('--probe_layers', type=str, default=None, help="自定义解冻层名列表，如 ['classifier','pre_classifier'] 或 classifier,pre_classifier")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if args.dataset_mode == 'imagenet100':
        num_labels = 100
    else:
        num_labels = 10
    model = create_encoder_model(
        args.model_name_or_path,
        num_labels=num_labels,
        pooling_strategy=args.pooling_strategy
    )

    # 关键修复：在内部的 transformer 模型上启用梯度检查点，避免 OOM
    if hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        print("✓ 已启用梯度检查点以避免 OOM")

    # LoRA 设置
    # LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj",
    #                "gate_proj","up_proj","down_proj"]
    # peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
    #                           inference_mode=False,
    #                           r=args.lora_r,
    #                           lora_alpha=args.lora_alpha,
    #                           target_modules=LORA_TARGET,
    #                           lora_dropout=0.1)
    # model = get_peft_model(model, peft_config)
    
    # 冻结和解冻指定层，可通过 --probe_layers 参数自定义
    import ast
    probe_layers = None
    if hasattr(args, 'probe_layers') and args.probe_layers:
        # 支持逗号分隔或 Python list 字符串
        try:
            probe_layers = ast.literal_eval(args.probe_layers)
            if not isinstance(probe_layers, list):
                probe_layers = [str(probe_layers)]
        except Exception:
            probe_layers = [x.strip() for x in args.probe_layers.split(',') if x.strip()]
    freeze_and_unfreeze(model, probe_layers)
    # 打印可训练参数
    print("可训练参数:")
    trainable_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
            trainable_count += param.numel()
    print(f"可训练参数总量: {trainable_count}")
    model = model.to(device)

    # 加载数据集
    if args.dataset_mode == 'imagenet100':
        import glob
        from PIL import Image
        import random
        # 获取所有类别文件夹
        class_dirs = sorted([d for d in os.listdir(args.imagenet100_dir) if os.path.isdir(os.path.join(args.imagenet100_dir, d))])
        class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}
        # 收集所有图片路径和标签
        all_samples = []
        for cls in class_dirs:
            img_dir = os.path.join(args.imagenet100_dir, cls)
            img_files = glob.glob(os.path.join(img_dir, '*.JPEG')) + glob.glob(os.path.join(img_dir, '*.jpg'))
            for img_path in img_files:
                all_samples.append({'image_path': img_path, 'label': class_to_idx[cls]})
        # 划分训练/测试
        random.seed(args.seed)
        random.shuffle(all_samples)
        train_samples = all_samples[:args.train_subset_size] if args.train_subset_size else all_samples
        test_samples = all_samples[-args.test_subset_size:] if args.test_subset_size else all_samples

        preprocess = create_preprocess_transform(args.image_size)
        def process_imagenet_sample(ex):
            img = Image.open(ex['image_path']).convert('RGB')
            ex_proc = {'image': img, 'label': ex['label']}
            out = tokenize_example_for_training(ex_proc, tokenizer, 'image', args.max_seq_len, preprocess)
            return out

        # 转为datasets格式
        from datasets import Dataset
        train_ds = Dataset.from_list(train_samples).map(
            process_imagenet_sample,
            batched=False,
            num_proc=12,
            desc="处理imagenet100训练数据"
        )
        test_ds = Dataset.from_list(test_samples).map(
            process_imagenet_sample,
            batched=False,
            num_proc=12,
            desc="处理imagenet100测试数据"
        )
    else:
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
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        eval_strategy='epoch',
        report_to=[] if args.disable_wandb else ['wandb'],
        run_name=None if args.disable_wandb else args.wandb_run_name,
    )

    def compute_metrics(p):
        preds, label_ids = p.predictions, p.label_ids
        preds = preds.argmax(-1)
        accuracy = (preds == label_ids).mean()
        
        # 简化的验证信息
        print(f"验证准确率: {accuracy:.4f}")
        
        return {'accuracy': accuracy}

    # 创建损失调整 Trainer，确保 wandb 记录调整后的损失
    class GradientAdjustedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            返回调整后的损失，确保 wandb 和日志都记录正确的损失
            """
            outputs = model(**inputs)
            
            # 处理不同的输出格式
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                raise ValueError(f"无法从模型输出中获取损失: {type(outputs)}")
            
            # 在训练时将损失除以梯度累积步数，使其与验证损失可比较
            if self.model.training and hasattr(self.args, 'gradient_accumulation_steps'):
                # 注意：这里直接调整返回的损失，会影响 wandb 记录
                adjusted_loss = loss / self.args.gradient_accumulation_steps
                return (adjusted_loss, outputs) if return_outputs else adjusted_loss
            else:
                # 验证时保持原损失
                return (loss, outputs) if return_outputs else loss

    # 动态padding到batch最大长度
    def collate_fn(batch):
        input_lens = [sum([1 for t in b['input_ids'] if t != tokenizer.pad_token_id and t is not None]) for b in batch]
        max_input_len = max(input_lens)
        input_ids = []
        attention_mask = []
        labels = []
        for b in batch:
            inp = b['input_ids'][:max_input_len]
            inp += [tokenizer.pad_token_id] * (max_input_len - len(inp))
            input_ids.append(inp)
            mask = b['attention_mask'][:max_input_len] if 'attention_mask' in b else [1]*len(inp)
            mask += [0] * (max_input_len - len(mask))
            attention_mask.append(mask)
            lab = b['labels']
            labels.append(lab)
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels_tensor = torch.tensor(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor
        }

    trainer = GradientAdjustedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(create_optimizer(), None),
        data_collator=collate_fn
    )

    # 开始训练
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)