#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example for MNIST: python /root/autodl-tmp/MLLM/evaluate_jpeglm.py --model_name_or_path /root/autodl-fs/models/jpeg-lm --checkpoint_dir /root/autodl-fs/trained_models/jpeglm/checkpoint-1000 --dataset mnist --batch_size 2 --test_subset_size 1000 --use_xformers --use_deepspeed --use_sdpa
# Example for CIFAR-10: python /root/autodl-tmp/MLLM/evaluate_jpeglm.py --model_name_or_path /root/autodl-fs/models/jpeg-lm --checkpoint_dir /root/autodl-tmp/MLLM/trained_models/jpeglm/jpeglm-cifar10 --dataset cifar10 --batch_size 2 --test_subset_size 1000 --use_xformers --use_deepspeed --use_sdpa

import argparse
import io
import torch
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

preprocess = transforms.Compose([transforms.RandomResizedCrop((256,256), scale=(1.0,1.0), ratio=(1.0,1.0), antialias=True)])
QUALITY = 25

def tokenize_example(example, tokenizer, max_seq_len, image_key):
    img = preprocess(example[image_key]) if preprocess else example[image_key].resize((256,256))
    img = img.convert("RGB")
    jpeg_str = convert_img_to_bytes(img, QUALITY)
    toks = tokenizer(jpeg_str, max_length=max_seq_len, padding="max_length", truncation=True)
    return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"], "labels": example["label"]}

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
    parser.add_argument("--use_xformers", action='store_true')
    parser.add_argument("--use_deepspeed", action='store_true')
    parser.add_argument("--use_sdpa", action='store_true', help="Enable PyTorch SDPA attention acceleration")
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Configure dataset-specific parameters
    if args.dataset == 'mnist':
        dataset_name = 'ylecun/mnist'
        max_seq_len = 2048
        image_key = 'image'
        num_labels = 10
    elif args.dataset == 'cifar10':
        dataset_name = 'uoft-cs/cifar10'
        max_seq_len = 3600
        image_key = 'img'
        num_labels = 10
    
    print(f"Evaluating on {args.dataset} dataset with max_seq_len={max_seq_len}")
    
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
    test = test.map(lambda ex: tokenize_example(ex, tokenizer, max_seq_len, image_key), 
                   batched=False, remove_columns=test.column_names)
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
