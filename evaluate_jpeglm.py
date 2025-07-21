#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example for MNIST: python /root/autodl-tmp/MLLM/evaluate_jpeglm.py --model_name_or_path /root/autodl-tmp/MLLM/models/jpeg-lm --checkpoint_dir /root/autodl-tmp/MLLM/trained_models/jpeglm/jpeglm-mnist-size96 --dataset mnist --batch_size 2 --test_subset_size 1000 --image_size 96 --use_sdpa --use_xformers --use_deepspeed --max_seq_len 2048
# Example for CIFAR-10: python /root/autodl-tmp/MLLM/evaluate_jpeglm.py --model_name_or_path //root/autodl-tmp/MLLM/models/jpeg-lm --checkpoint_dir /root/autodl-tmp/MLLM/trained_models/jpeglm/jpeglm-cifar10 --dataset cifar10 --batch_size 2 --test_subset_size 1000 --image_size 256 --use_sdpa --use_xformers --use_deepspeed --max_seq_len 2048
# Example with bit flip: python /root/autodl-tmp/MLLM/evaluate_jpeglm.py --model_name_or_path /root/autodl-tmp/MLLM/models/jpeg-lm --checkpoint_dir /root/autodl-tmp/MLLM/trained_models/jpeglm/jpeglm-cifar10 --dataset cifar10 --batch_size 2 --test_subset_size 1000 --image_size 256 --bit_flip --bit_flip_prob 0.001 --max_seq_len 2048

import argparse
import io
import torch
import sys
import os
# 添加utils路径以导入统一的数据处理工具
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from data_utils import tokenize_example_for_evaluation, create_preprocess_transform

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, set_seed
from peft import PeftModel
from torchvision import transforms

def convert_img_to_bytes(img: Image.Image, quality: int):
    img.save("cache_tables.jpg", format="JPEG", quality=quality, subsampling="4:2:0", streamtype=1, restart_marker_blocks=1)
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality, subsampling="4:2:0", streamtype=2, restart_marker_blocks=1)
        data = buf.getvalue()
    return ''.join(chr(b + 10240) for b in data)


def collate_fn(batch):
    ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    masks = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": masks, "labels": labels}

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset", choices=['mnist', 'cifar10'], default='mnist', help="Dataset to evaluate on (mnist or cifar10)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_subset_size", type=int)
    parser.add_argument("--image_size", type=int, default=256, help="Image side length for preprocessing (e.g. 96 or 256)")
    parser.add_argument("--use_xformers", action='store_true')
    parser.add_argument("--use_deepspeed", action='store_true')
    parser.add_argument("--use_sdpa", action='store_true', help="Enable PyTorch SDPA attention acceleration")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum token sequence length override")
    parser.add_argument("--bit_flip", action='store_true', help="Enable bit flip corruption for JPEG data")
    parser.add_argument("--bit_flip_prob", type=float, default=0.001, help="Probability of flipping each bit (default: 0.001, i.e., 0.1%)")
    args = parser.parse_args()
    set_seed(args.seed)
    # Override max_seq_len if provided
    if args.max_seq_len is not None:
        max_seq_len = args.max_seq_len

    # Configure dataset-specific parameters (unless overridden)
    if args.dataset == 'mnist':
        dataset_name = 'ylecun/mnist'
        if args.max_seq_len is None:
            max_seq_len = 2048
        image_key = 'image'
        num_labels = 10
    elif args.dataset == 'cifar10':
        dataset_name = 'uoft-cs/cifar10'
        if args.max_seq_len is None:
            max_seq_len = 3600
        image_key = 'img'
        num_labels = 10

    print(f"Evaluating on {args.dataset} dataset with max_seq_len={max_seq_len}, image_size={args.image_size}")
    
    # 输出比特反转状态信息
    if args.bit_flip:
        print(f"启用比特反转模式，反转概率: {args.bit_flip_prob:.6f} ({args.bit_flip_prob*100:.4f}%)")
    else:
        print("未启用比特反转模式")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, use_cache=False)
    base_kwargs = dict()
    if args.use_sdpa:
        base_kwargs['attn_implementation'] = 'sdpa'
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, **base_kwargs)
    model = PeftModel.from_pretrained(base, args.checkpoint_dir).to(device).eval()
    if args.use_xformers:
        try:
            model.enable_xformers_memory_efficient_attention()
        except:
            pass
    if args.use_deepspeed:
        try:
            import deepspeed
            model = deepspeed.init_inference(model, mp_size=1, dtype=torch.half, replace_with_kernel_inject=True)
        except:
            pass
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    model.half()
    
    # 设置动态preprocess
    preprocess = create_preprocess_transform(args.image_size)

    # Load dataset
    ds = load_dataset(dataset_name)

    # Handle different dataset structures
    if args.dataset == 'mnist':
        ds = ds.cast_column('image', ds['train'].features['image'])
        test = ds['test']
    elif args.dataset == 'cifar10':
        test = ds['test']

    if args.test_subset_size:
        test = test.select(range(min(args.test_subset_size, len(test))))

    # Apply tokenization with dataset-specific parameters
    bit_flip_prob = args.bit_flip_prob if args.bit_flip else 0.0
    test = test.map(lambda ex: tokenize_example_for_evaluation(ex, tokenizer, max_seq_len, image_key, preprocess, bit_flip_prob=bit_flip_prob), 
                   batched=False, remove_columns=test.column_names, num_proc=4)
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    total = len(test)
    correct = processed = 0
    class_correct = {i: 0 for i in range(num_labels)}
    class_total = {i: 0 for i in range(num_labels)}
    pred_count = {i: 0 for i in range(num_labels)}
    print(f"Total {total} samples. Start evaluation on {args.dataset} dataset.")
    print("Press Ctrl+C to interrupt and display progress\n")
    try:
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.inference_mode(), torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)
            for t, p in zip(labels.tolist(), preds.tolist()):
                processed += 1
                class_total[t] += 1
                pred_count[p] += 1
                if p == t:
                    correct += 1
                    class_correct[t] += 1
            current_acc = correct / processed
            tqdm.write(f"[{processed}/{total}] True={t} Pred={p} Acc={correct}/{processed}={current_acc:.4f}")
            class_accs = "  ".join(f"{i}:{class_correct[i]}/{class_total[i]}={class_correct[i]/class_total[i] if class_total[i]>0 else 0.0:.3f}" for i in range(num_labels))
            pred_dist = "  ".join(f"{i}:{pred_count[i]}/{processed}={pred_count[i]/processed:.3f}" for i in range(num_labels))
            tqdm.write("Class Accs: " + class_accs)
            tqdm.write("Pred Dist: " + pred_dist + "\n")
    except KeyboardInterrupt:
        print("\nInterrupted, showing current results…")
    if processed:
        final_acc = correct / processed
        print(f"Final Accuracy on {args.dataset}: {correct}/{processed} = {final_acc:.4f}")
        for i in range(num_labels):
            acc_i = class_correct[i] / class_total[i] if class_total[i] else 0.0
            print(f"  Class {i}: {class_correct[i]}/{class_total[i]} = {acc_i:.4f}")
    else:
        print("No samples processed.")
