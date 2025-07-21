#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image Captioning Inference Script
加载训练好的 LoRA 检查点，对指定图像生成描述
"""

import argparse
import torch
import sys
from PIL import Image
from transformers import AutoTokenizer
from peft import PeftModel

# 添加路径
sys.path.append('jpeg-lm/models')
from jpeglm_encoder import create_seq2seq_model
sys.path.append('utils')
from data_utils import convert_img_to_bytes, create_preprocess_transform

def parse_args():
    parser = argparse.ArgumentParser(description="Image Captioning Inference")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="训练好的模型检查点目录")
    parser.add_argument("--image_path", type=str, required=True,
                        help="输入图像路径")
    parser.add_argument("--encoder_model", type=str, default="/root/autodl-tmp/MLLM/models/jpeg-lm",
                        help="编码器模型路径")
    parser.add_argument("--decoder_model", type=str, default="gpt2",
                        help="解码器模型路径")
    parser.add_argument("--image_size", type=int, default=96,
                        help="图像预处理尺寸")
    parser.add_argument("--max_enc_len", type=int, default=2048,
                        help="编码器最大序列长度")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大新token数")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="nucleus sampling 参数")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="beam search 数量")
    return parser.parse_args()

def load_model_and_tokenizers(args):
    """加载模型和tokenizers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizers
    enc_tok = AutoTokenizer.from_pretrained(f"{args.checkpoint_dir}/enc_tok", use_fast=False)
    dec_tok = AutoTokenizer.from_pretrained(f"{args.checkpoint_dir}/dec_tok", use_fast=False)
    
    # 创建基础模型
    base_model = create_seq2seq_model(
        args.encoder_model,
        args.decoder_model,
        encoder_pooling_strategy='mean'
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model = model.to(device)
    model.eval()
    
    return model, enc_tok, dec_tok, device

def preprocess_image(image_path, enc_tok, preprocess, max_enc_len):
    """预处理图像"""
    # 加载和预处理图像
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img)
    
    # 转换为JPEG字符串
    jpeg_str = convert_img_to_bytes(img)
    
    # Tokenize
    enc_ids = enc_tok(jpeg_str,
                      add_special_tokens=False,
                      max_length=max_enc_len-1,
                      truncation=True)["input_ids"]
    
    # 确保token ID在有效范围内
    enc_vocab_size = enc_tok.vocab_size
    enc_ids = [min(token_id, enc_vocab_size-1) for token_id in enc_ids]
    
    # 添加BOS token并padding
    enc_ids = [enc_tok.bos_token_id] + enc_ids[:max_enc_len-1]
    if len(enc_ids) < max_enc_len:
        enc_ids += [enc_tok.pad_token_id] * (max_enc_len - len(enc_ids))
    
    # 创建attention mask
    enc_mask = [1 if i != enc_tok.pad_token_id else 0 for i in enc_ids]
    
    return torch.tensor([enc_ids]), torch.tensor([enc_mask])

def generate_caption(model, input_ids, attention_mask, dec_tok, args, device):
    """生成图像描述"""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # 生成参数
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "early_stopping": True,
        "pad_token_id": dec_tok.pad_token_id,
        "eos_token_id": dec_tok.eos_token_id,
        "do_sample": True if args.temperature > 0 else False,
    }
    
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    
    # 解码生成的文本
    generated_text = dec_tok.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def main():
    args = parse_args()
    
    print(f"加载模型从: {args.checkpoint_dir}")
    print(f"输入图像: {args.image_path}")
    
    # 加载模型和tokenizers
    model, enc_tok, dec_tok, device = load_model_and_tokenizers(args)
    
    # 创建图像预处理器
    preprocess = create_preprocess_transform(args.image_size)
    
    # 预处理图像
    print("预处理图像...")
    input_ids, attention_mask = preprocess_image(
        args.image_path, enc_tok, preprocess, args.max_enc_len
    )
    
    # 生成描述
    print("生成图像描述...")
    caption = generate_caption(
        model, input_ids, attention_mask, dec_tok, args, device
    )
    
    print(f"\n生成的图像描述: {caption}")

if __name__ == "__main__":
    main()
