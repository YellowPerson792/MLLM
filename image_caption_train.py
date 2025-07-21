#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flickr8K Image Captioning Training Script — 修正版 A
- 单独使用 JPEG-LM 的 tokenizer 处理图像码流
- 单独使用 GPT‑2 的 tokenizer 处理文本 caption
- 手动恢复 -100 为 pad_token_id，再右移
"""
# python /root/autodl-tmp/MLLM/image_caption_train.py     --encoder_model /root/autodl-tmp/MLLM/models/jpeg-lm     --decoder_model gpt2     --output_dir /root/autodl-tmp/MLLM/checkpoints/jpeglm-gpt2-captioning     --image_size 96     --max_enc_len 2048     --max_dec_len 64     --batch_size 1     --gradient_accumulation_steps 1     --epochs 3     --learning_rate 2e-4     --lora_r 8     --lora_alpha 32     --fp16     --seed 42     --disable_wandb

import argparse, os, sys, torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from peft import get_peft_model, LoraConfig, TaskType
sys.path.append('jpeg-lm/models')
from jpeglm_encoder import create_seq2seq_model
sys.path.append('utils')
from data_utils import convert_img_to_bytes, create_preprocess_transform

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_model", required=True)
    p.add_argument("--decoder_model", default="gpt2")
    p.add_argument("--output_dir", type=str, default="./output", help="模型输出目录")
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--max_enc_len", type=int, default=1024)
    p.add_argument("--max_dec_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--fp16", action='store_true')
    p.add_argument("--disable_wandb", action='store_true')
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def preprocess_dataset(ex, enc_tok, dec_tok, prep, max_enc, max_dec):
    # ---- 图像编码 ----
    img = prep(ex['image']).convert('RGB')
    jpeg_str = convert_img_to_bytes(img)
    enc_ids = enc_tok(jpeg_str,
                      add_special_tokens=False,
                      max_length=max_enc-1,
                      truncation=True)["input_ids"]
    
    # 确保token ID在有效范围内
    enc_vocab_size = enc_tok.vocab_size
    enc_ids = [min(token_id, enc_vocab_size-1) for token_id in enc_ids]
    
    enc_ids = [enc_tok.bos_token_id] + enc_ids[:max_enc-1]
    if len(enc_ids) < max_enc:
        enc_ids += [enc_tok.pad_token_id] * (max_enc - len(enc_ids))
    enc_mask = [1 if i!=enc_tok.pad_token_id else 0 for i in enc_ids]

    # ---- 文本标签 ----
    # jxie/flickr8k 数据集格式：caption_0, caption_1, caption_2, caption_3, caption_4
    import random
    
    available_captions = []
    for i in range(5):  # caption_0 到 caption_4
        caption_key = f'caption_{i}'
        if caption_key in ex and ex[caption_key]:
            caption_text = ex[caption_key].strip()
            if caption_text:  # 非空 caption
                available_captions.append(caption_text)
    
    # 选择一个可用的 caption
    if available_captions:
        cap = random.choice(available_captions)
    else:
        cap = "A photo."  # 默认描述
    
    dec = dec_tok(cap,
                  max_length=max_dec,
                  padding='max_length',
                  truncation=True,
                  return_tensors=None)
    
    # labels 用 -100，训练时才会被 ignore
    labels = [tok if tok!=dec_tok.pad_token_id else -100 
              for tok in dec['input_ids']]

    return {"input_ids": enc_ids,
            "attention_mask": enc_mask,
            "labels": labels}

def shift_labels_to_inputs(labels, bos, pad):
    # 把 -100 恢复回 pad，右移
    raw = [pad if x==-100 else x for x in labels]
    # 右移
    out = [bos] + raw[:-1]
    return out

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # **分别加载两套 tokenizer**
    enc_tok = AutoTokenizer.from_pretrained(args.encoder_model, use_fast=False)
    dec_tok = AutoTokenizer.from_pretrained(args.decoder_model, use_fast=False)
    
    # 强制正确设置 pad_token
    if dec_tok.pad_token is None:
        dec_tok.add_special_tokens({"pad_token": "[PAD]"})
        dec_tok.pad_token = "[PAD]"
        
    # 对于 GPT-2，通常使用 eos_token 作为 pad_token
    if "gpt2" in args.decoder_model.lower():
        dec_tok.pad_token = dec_tok.eos_token
    
    # 验证 pad_token_id
    if dec_tok.pad_token_id is None:
        dec_tok.pad_token_id = dec_tok.eos_token_id

    # 模型
    model = create_seq2seq_model(args.encoder_model,
                                 args.decoder_model,
                                 encoder_pooling_strategy='mean')
    # 开启梯度检查点
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
    if hasattr(model.decoder, 'gradient_checkpointing_enable'):
        model.decoder.gradient_checkpointing_enable()
    model.to(device)

    # LoRA 配置
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r, 
        lora_alpha=args.lora_alpha,
        target_modules=[
            # Llama 编码器层
            "q_proj", "k_proj", "v_proj", "o_proj",  
            "gate_proj", "up_proj", "down_proj",     
            # GPT-2 解码器层
            "c_attn", "c_proj",                      
            "c_fc",                                  
            # GPT-2 cross-attention 层
            "q_attn",                                
        ],
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_cfg).to(device)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()

    # 数据
    ds = load_dataset('jxie/flickr8k',split='train').train_test_split(0.1,seed=args.seed)
    prep = create_preprocess_transform(args.image_size)
    # Tokenize
    train = ds['train'].map(lambda ex: preprocess_dataset(ex, enc_tok, dec_tok, prep,
                                                          args.max_enc_len, args.max_dec_len),
                            remove_columns=ds['train'].column_names, num_proc=12)
    val   = ds['test' ].map(lambda ex: preprocess_dataset(ex, enc_tok, dec_tok, prep,
                                                          args.max_enc_len, args.max_dec_len),
                            remove_columns=ds['test'].column_names, num_proc=12)

    # collate_fn
    def collate_fn(batch):
        enc_ids = torch.tensor([b['input_ids'] for b in batch])
        enc_mask= torch.tensor([b['attention_mask'] for b in batch])
        labs    = [b['labels'] for b in batch]
        
        labels_tensor = torch.tensor(labs)
        
        return {
            "input_ids":      enc_ids,
            "attention_mask": enc_mask,
            "labels":         labels_tensor
        }

    # Trainer
    args_tr = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy='steps',
        eval_steps=args.save_steps,
        fp16=args.fp16,
        seed=args.seed,
        report_to=[] if args.disable_wandb else ['wandb'],
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args_tr,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collate_fn,
        tokenizer=dec_tok,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    enc_tok.save_pretrained(args.output_dir+"/enc_tok")
    dec_tok.save_pretrained(args.output_dir+"/dec_tok")

if __name__=='__main__':
    main()
