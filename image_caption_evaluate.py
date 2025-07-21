#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 示例运行命令：
# python /root/autodl-tmp/MLLM/image_caption_evaluate.py --checkpoint_dir /root/autodl-fs/trained_models/jpeglm/jpeglm-gpt2-v1 --encoder_model /root/autodl-fs/models/jpeg-lm --decoder_model gpt2 --dataset jxie/flickr8k --split test --image_size 96 --max_enc_len 2048 --max_new_tokens 50 --num_beams 5 --num_samples 100 --output_dir ./eval_results --batch_size 2
"""
Simple Image Captioning Evaluation Script
简化版评估脚本，不依赖外部评估包
"""

import argparse
import torch
import sys
import os
import json
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel
import numpy as np

# 添加路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jpeg-lm/models'))
from jpeglm_encoder import create_seq2seq_model
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from data_utils import convert_img_to_bytes, create_preprocess_transform

def simple_bleu_score(reference, candidate, n=4):
    """简单的BLEU分数计算"""
    from collections import Counter
    import math
    
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # 计算n-gram精度
    precisions = []
    for i in range(1, n+1):
        ref_ngrams = Counter([tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1)])
        cand_ngrams = Counter([tuple(cand_tokens[j:j+i]) for j in range(len(cand_tokens)-i+1)])
        
        if len(cand_ngrams) == 0:
            precisions.append(0)
        else:
            matches = sum((ref_ngrams & cand_ngrams).values())
            precisions.append(matches / len(cand_ngrams))
    
    # 简化的BLEU计算
    if min(precisions) > 0:
        log_precisions = [math.log(p) for p in precisions]
        geometric_mean = math.exp(sum(log_precisions) / len(log_precisions))
        
        # 简化的brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(cand_tokens)))
        
        return bp * geometric_mean
    else:
        return 0.0

def evaluate_simple(args):
    """简化评估函数"""
    print(f"加载模型从: {args.checkpoint_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和tokenizers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载tokenizers
    enc_tok = AutoTokenizer.from_pretrained(args.encoder_model, use_fast=False)
    dec_tok = AutoTokenizer.from_pretrained(args.decoder_model, use_fast=False)
    # 自动修正 pad_token_id，避免警告
    if dec_tok.pad_token_id is None:
        dec_tok.pad_token_id = dec_tok.eos_token_id

    # 创建基础模型
    base_model = create_seq2seq_model(
        args.encoder_model,
        args.decoder_model,
        encoder_pooling_strategy='mean'
    )

    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    # 半精度推理
    if torch.cuda.is_available():
        model = model.half()
    model = model.to(device)
    model.eval()
    # 已关闭SDPA加速，使用默认attention实现

    # 创建图像预处理器
    preprocess = create_preprocess_transform(args.image_size)
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args.dataset)
    eval_split = dataset[args.split]
    
    # 限制评估样本数量
    if args.num_samples is not None:
        eval_split = eval_split.select(range(min(args.num_samples, len(eval_split))))
    
    print(f"评估样本数量: {len(eval_split)}")
    
    # 预处理阶段
    batch_size = getattr(args, "batch_size", 8)
    all_input_ids, all_attention_mask, all_references = [], [], []
    for sample in tqdm(eval_split, desc="预处理"):
        # 预处理图像
        img = preprocess(sample['image'])
        jpeg_str = convert_img_to_bytes(img)
        
        # Tokenize
        enc_ids = enc_tok(jpeg_str,
                          add_special_tokens=False,
                          max_length=args.max_enc_len-1,
                          truncation=True)["input_ids"]
        
        # 确保token ID在有效范围内
        enc_vocab_size = enc_tok.vocab_size
        enc_ids = [min(token_id, enc_vocab_size-1) for token_id in enc_ids]
        
        # 添加BOS token并padding
        enc_ids = [enc_tok.bos_token_id] + enc_ids[:args.max_enc_len-1]
        if len(enc_ids) < args.max_enc_len:
            enc_ids += [enc_tok.pad_token_id] * (args.max_enc_len - len(enc_ids))
        
        # 创建attention mask
        enc_mask = [1 if token_id != enc_tok.pad_token_id else 0 for token_id in enc_ids]
        
        all_input_ids.append(enc_ids)
        all_attention_mask.append(enc_mask)
        
        # 获取参考答案
        refs = []
        for j in range(5):
            key = f'caption_{j}'
            if key in sample and sample[key]:
                refs.append(sample[key].strip())
        if not refs:
            refs = ["A photo."]
        all_references.append(refs)

    bleu_scores = []
    exact_matches = 0
    total_samples = len(all_input_ids)
    print("开始批量推理评估...")

    # 推理阶段
    for start in tqdm(range(0, total_samples, batch_size), desc="推理"):
        end = min(start + batch_size, total_samples)
        batch_input_ids = torch.tensor(all_input_ids[start:end]).to(device)
        batch_attention_mask = torch.tensor(all_attention_mask[start:end]).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=dec_tok.pad_token_id,
                eos_token_id=dec_tok.eos_token_id,
                do_sample=False
            )
        for i, output in enumerate(outputs):
            generated_text = dec_tok.decode(output, skip_special_tokens=True).strip()
            references = all_references[start + i]
            best_bleu = max([simple_bleu_score(ref, generated_text) for ref in references])
            bleu_scores.append(best_bleu)
            if generated_text.lower() in [ref.lower() for ref in references]:
                exact_matches += 1
        if (start // batch_size + 1) % 2 == 0:
            avg_bleu = np.mean(bleu_scores)
            print(f"已处理 {end} 样本，平均BLEU: {avg_bleu:.4f}")
    
    # 计算最终指标
    final_bleu = np.mean(bleu_scores)
    exact_match_rate = exact_matches / total_samples
    
    print("\n" + "="*50)
    print("简化评估结果:")
    print("="*50)
    print(f"平均 BLEU-4: {final_bleu:.4f}")
    print(f"精确匹配率: {exact_match_rate:.4f}")
    print(f"评估样本数: {total_samples}")
    
    # 保存结果
    results = {
        "bleu_score": final_bleu,
        "exact_match_rate": exact_match_rate,
        "num_samples": total_samples,
        "checkpoint": args.checkpoint_dir,
        "dataset": args.dataset,
        "split": args.split
    }
    
    results_path = os.path.join(args.output_dir, "simple_evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Image Captioning Evaluation")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--encoder_model", type=str, default="/root/autodl-tmp/MLLM/models/jpeg-lm")
    parser.add_argument("--decoder_model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="jxie/flickr8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--max_enc_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--batch_size", type=int, default=8, help="推理批量大小")
    args = parser.parse_args()
    evaluate_simple(args)