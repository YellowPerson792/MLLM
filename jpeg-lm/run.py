#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example: python eval_mnist_checkpoint.py --model_name_or_path /path/to/model --checkpoint_dir /path/to/checkpoint --batch_size 2 --test_subset_size 1000 --use_xformers --use_deepspeed
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

preprocess = transforms.Compose([
    transforms.RandomResizedCrop((256,256), scale=(1.0,1.0), ratio=(1.0,1.0), antialias=True)
])

QUALITY = 25
MAX_SEQ_LEN = 2048

def tokenize_example(example, tokenizer):
    img = preprocess(example["image"]) if preprocess else example["image"].resize((256,256))
    img = img.convert("RGB")
    jpeg_str = convert_img_to_bytes(img, QUALITY)
    toks = tokenizer(jpeg_str, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True)
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_subset_size", type=int)
    parser.add_argument("--use_xformers", action='store_true')
    parser.add_argument("--use_deepspeed", action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=10, use_cache=False)
    base = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model = PeftModel.from_pretrained(base, args.checkpoint_dir).to(device).eval()

    if args.use_xformers:
        model.enable_xformers_memory_efficient_attention()
    if args.use_deepspeed:
        import deepspeed
        model = deepspeed.init_inference(model, mp_size=1, dtype=torch.half, replace_with_kernel_inject=True)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    model.half()

    ds = load_dataset('ylecun/mnist')
    ds = ds.cast_column('image', ds['train'].features['image'])
    test = ds['test']
    if args.test_subset_size:
        test = test.select(range(args.test_subset_size))
    test = test.map(lambda ex: tokenize_example(ex, tokenizer), batched=False, remove_columns=test.column_names)
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    total = len(test)
    correct = processed = 0
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    pred_count = {i: 0 for i in range(10)}

    print(f"Start evaluation on {total} samples (InferenceMode + AMP + compile + cuDNN)")
    print("Press Ctrl+C to interrupt and show current results\n")
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
            class_accs = "  ".join(f"{i}:{class_correct[i]}/{class_total[i]}={class_correct[i]/class_total[i]:.3f}" for i in range(10))
            tqdm.write("  Class Accs: " + class_accs)
            pred_dist = "  ".join(f"{i}:{pred_count[i]}/{processed}={pred_count[i]/processed:.3f}" for i in range(10))
            tqdm.write("  Pred Dist: " + pred_dist + "\n")
    except KeyboardInterrupt:
        print("\nInterrupted, showing current results…")

    if processed:
        final_acc = correct / processed
        print(f"Final Accuracy: {correct}/{processed} = {final_acc:.4f}")
        for i in range(10):
            print(f"Class {i}: {class_correct[i]}/{class_total[i]} = {class_correct[i]/class_total[i]:.4f}")
    else:
        print("No samples processed.")
