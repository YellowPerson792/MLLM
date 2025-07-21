#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 示例运行命令：
# 随机选择: python /root/autodl-tmp/MLLM/image_caption_inference.py --checkpoint_dir /root/autodl-fs/trained_models/jpeglm/jpeglm-gpt2-v1 --encoder_model /root/autodl-fs/models/jpeg-lm --decoder_model gpt2 --image_size 96 --max_enc_len 2048 --max_new_tokens 50 --top_p 0.9 --top_k 50 
# 指定样本: python /root/autodl-tmp/MLLM/image_caption_inference.py --checkpoint_dir /root/autodl-fs/trained_models/jpeglm/jpeglm-gpt2-v1 --encoder_model /root/autodl-fs/models/jpeg-lm --decoder_model gpt2 --image_size 96 --max_enc_len 2048 --max_new_tokens 50 --top_p 0.9 --top_k 50 --image_path /root/autodl-tmp/MLLM/datasets/flickr8k/images/sample_338_example.jpg
"""
Image Captioning Inference Script
从Flickr8K测试集随机选择图片或指定图片，显示原始captions，保存图片，然后执行推理生成描述
"""

import argparse
import torch
import sys
import random
import os
from PIL import Image
from transformers import AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# 添加路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jpeg-lm/models'))
from jpeglm_encoder import create_seq2seq_model
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
from data_utils import convert_img_to_bytes, create_preprocess_transform

def parse_args():
    parser = argparse.ArgumentParser(description="Image Captioning Inference from Flickr8K Test Set")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="训练好的模型检查点目录")
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
    parser.add_argument("--top_k", type=int, default=50,
                        help="top-k 采样参数")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="beam search 数量")
    # 已移除 --seed 参数
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="指定要推理的样本索引，如果不指定则随机选择")
    parser.add_argument("--image_path", type=str, default=None,
                        help="指定任意图片路径进行推理（优先级最高）")
    parser.add_argument("--save_dir", type=str, default="/root/autodl-tmp/MLLM/datasets/flickr8k/images",
                        help="保存图片的目录")
    return parser.parse_args()

def load_flickr8k_test_set():
    """加载Flickr8K测试集"""
    print("正在加载Flickr8K数据集...")
    dataset = load_dataset("jxie/flickr8k", split="test")
    print(f"测试集包含 {len(dataset)} 张图片")
    return dataset

def select_sample(dataset, sample_idx=None, seed=42, save_dir=None):
    """从数据集中选择一个样本（随机或指定索引）"""
    if sample_idx is not None:
        # 指定索引
        if sample_idx < 0 or sample_idx >= len(dataset):
            raise ValueError(f"样本索引 {sample_idx} 超出范围 [0, {len(dataset)-1}]")
        selected_idx = sample_idx
        print(f"\n=== 指定的样本 (索引: {selected_idx}) ===")
    else:
        # 真随机选择（不设置seed）
        selected_idx = random.randint(0, len(dataset) - 1)
        print(f"\n=== 随机选择的样本 (索引: {selected_idx}) ===")
    sample = dataset[selected_idx]
    print(f"图片ID: {sample.get('image_id', 'N/A')}")
    # 仅在随机选择时显示原始caption
    if sample_idx is None:
        print(f"\n原始的5条captions:")
        for i in range(5):
            key = f'caption_{i}'
            caption = sample.get(key, None)
            if caption:
                print(f"  {i+1}. {caption}")
            else:
                print(f"  {i+1}. [无内容]")
    # 保存图片
    if save_dir is not None:
        save_image(sample['image'], selected_idx, save_dir)
    return sample, selected_idx

def save_image(pil_image, sample_idx, save_dir):
    """保存图片到指定目录，覆盖之前的图片"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 固定的文件名，每次都会覆盖
    image_filename = "current_sample.jpg"
    image_path = os.path.join(save_dir, image_filename)
    
    # 保存图片
    pil_image.save(image_path, "JPEG", quality=95)
    print(f"\n图片已保存到: {image_path}")
    print(f"图片来源: 测试集样本索引 {sample_idx}")

def load_model_and_tokenizers(args):
    """加载模型和tokenizers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizers - 与训练脚本保持一致
    enc_tok = AutoTokenizer.from_pretrained(args.encoder_model, use_fast=False)
    dec_tok = AutoTokenizer.from_pretrained(args.decoder_model, use_fast=False)
    
    # 强制正确设置 pad_token - 与训练脚本保持一致
    if dec_tok.pad_token is None:
        dec_tok.add_special_tokens({"pad_token": "[PAD]"})
        dec_tok.pad_token = "[PAD]"
        
    # 对于 GPT-2，通常使用 eos_token 作为 pad_token
    if "gpt2" in args.decoder_model.lower():
        dec_tok.pad_token = dec_tok.eos_token
    
    # 验证 pad_token_id
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
    # SDPA加速（仅支持PyTorch 2.0+，自动开启）
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("✓ 已启用SDPA加速（scaled_dot_product_attention）")
    except Exception as e:
        print(f"⚠️ SDPA加速未启用: {e}")
    return model, enc_tok, dec_tok, device

def preprocess_image(pil_image, enc_tok, preprocess, max_enc_len):
    """预处理图像 - 与训练脚本保持一致"""
    # 确保图像是RGB格式
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # 应用预处理
    img = preprocess(pil_image)
    
    # 转换为JPEG字符串
    jpeg_str = convert_img_to_bytes(img)
    
    # Tokenize - 与训练脚本保持一致
    enc_ids = enc_tok(jpeg_str,
                      add_special_tokens=False,
                      max_length=max_enc_len-1,
                      truncation=True)["input_ids"]
    
    # 确保token ID在有效范围内 - 与训练脚本保持一致
    enc_vocab_size = enc_tok.vocab_size
    enc_ids = [min(token_id, enc_vocab_size-1) for token_id in enc_ids]
    
    # 添加BOS token并padding - 与训练脚本保持一致
    enc_ids = [enc_tok.bos_token_id] + enc_ids[:max_enc_len-1]
    if len(enc_ids) < max_enc_len:
        enc_ids += [enc_tok.pad_token_id] * (max_enc_len - len(enc_ids))
    
    # 创建attention mask - 与训练脚本保持一致
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
        "max_new_tokens": min(args.max_new_tokens, 512),  # 限制最大生成长度
        "temperature": max(args.temperature, 0.1),  # 避免温度为0导致的问题
        "top_p": min(max(args.top_p, 0.1), 1.0),  # 确保top_p在有效范围
        "top_k": max(args.top_k, 1),  # top-k采样
        "num_beams": max(args.num_beams, 1),  # 确保beam数量至少为1
        "early_stopping": True,
        "pad_token_id": dec_tok.pad_token_id if dec_tok.pad_token_id is not None else dec_tok.eos_token_id,
        "eos_token_id": dec_tok.eos_token_id,
        "do_sample": True if args.temperature > 0 or args.top_p < 1.0 or args.top_k > 1 else False,
    }
    
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    
    # 解码生成的文本
    generated_text = dec_tok.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def main():
    args = parse_args()
    
    print(f"加载模型从: {args.checkpoint_dir}")
    # 加载模型和tokenizers
    print(f"\n正在加载模型和tokenizers...")
    model, enc_tok, dec_tok, device = load_model_and_tokenizers(args)
    # 创建图像预处理器
    preprocess = create_preprocess_transform(args.image_size)
    if args.image_path is not None:
        # 推理任意路径图片
        print(f"\n使用指定图片路径进行推理: {args.image_path}")
        pil_image = Image.open(args.image_path)
        print(f"图像尺寸: {pil_image.size}")
        print(f"图像模式: {pil_image.mode}")
        print("\n正在预处理图像...")
        input_ids, attention_mask = preprocess_image(
            pil_image, enc_tok, preprocess, args.max_enc_len
        )
        print(f"输入token长度: {attention_mask.sum().item()}")
        print("\n正在生成图像描述...")
        caption = generate_caption(
            model, input_ids, attention_mask, dec_tok, args, device
        )
        print(f"\n" + "="*60)
        print(f"推理结果:")
        print(f"图片路径: {args.image_path}")
        print(f"生成的图像描述: {caption}")
        print(f"="*60)
    else:
        # 兼容原有流程（测试集）
        if args.sample_idx is not None:
            print(f"指定样本索引: {args.sample_idx}")
        # 加载Flickr8K测试集
        test_dataset = load_flickr8k_test_set()
        # 选择样本（随机或指定索引）
        sample, selected_idx = select_sample(
            test_dataset,
            sample_idx=args.sample_idx,
            save_dir=args.save_dir
        )
        # 获取图像
        pil_image = sample['image']
        print(f"\n图像尺寸: {pil_image.size}")
        print(f"图像模式: {pil_image.mode}")
        print("\n正在预处理图像...")
        input_ids, attention_mask = preprocess_image(
            pil_image, enc_tok, preprocess, args.max_enc_len
        )
        print(f"输入token长度: {attention_mask.sum().item()}")
        print("\n正在生成图像描述...")
        caption = generate_caption(
            model, input_ids, attention_mask, dec_tok, args, device
        )
        print(f"\n" + "="*60)
        print(f"推理结果:")
        print(f"样本索引: {selected_idx}")
        print(f"生成的图像描述: {caption}")
        print(f"="*60)

if __name__ == "__main__":
    main()
