# 示例运行命令：
# python /root/autodl-tmp/MLLM/ImageCaption/train_prefixlm_jpeglm_gpt2_cls.py \
#   --output_dir ./checkpoints/prefixlm-jpeglm-gpt2-mnist-classification-lora \
#   --train_batch_size 2 \
#   --eval_batch_size 2 \
#   --eval_strategy steps \
#   --eval_steps 128 \
#   --logging_steps 64 \
#   --save_steps 512 \
#   --warmup_steps 512 \
#   --learning_rate 2e-4 \
#   --num_train_epochs 3 \
#   --save_total_limit 6 \
#   --lr_scheduler_type linear \
#   --gradient_accumulation_steps 8 \
#   --report_to none \
#   --bf16 \
#   --max_length 1024 \
#   --image_size 96 \
#   --num_train_samples 6000 \
#   --num_eval_samples 16

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, GenerationConfig
from jpeglm.models.jpeglm_encoder import create_jpeglm_encoder_with_pooling
from utils.data_utils import convert_img_to_bytes, create_preprocess_transform
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="./checkpoints/prefixlm-jpeglm-gpt2-mnist-classification")
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--image_size', type=int, default=28)
parser.add_argument('--max_length', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--num_train_samples', type=int, default=6000)
parser.add_argument('--num_eval_samples', type=int, default=1000)
parser.add_argument('--eval_strategy', type=str, default="epoch")
parser.add_argument('--eval_steps', type=int, default=200)
parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--save_steps', type=int, default=200)
parser.add_argument('--save_total_limit', type=int, default=2)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--lr_scheduler_type', type=str, default="linear")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--report_to', type=str, default=None)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--bf16', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST数字标签映射到文本
digit_to_text = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}
tiny_vocab = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
word2id = {w: i for i, w in enumerate(tiny_vocab)}
id2word = {i: w for i, w in enumerate(tiny_vocab)}

# 加载tokenizer和模型
encoder_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/MLLM/models/jpeg-lm")
decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")
decoder_tokenizer.pad_token = decoder_tokenizer.unk_token

# 将tiny_vocab中的单词转换为GPT2 tokenizer中的token IDs
gpt2_token_ids = {}
for word in tiny_vocab:
    token_ids = decoder_tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) == 1:
        gpt2_token_ids[word] = token_ids[0]
    else:
        print(f"警告: '{word}' 被tokenize为多个token: {token_ids}")
        gpt2_token_ids[word] = token_ids[0]  # 取第一个token

print("数字词汇到GPT2 token ID的映射:")
for word, token_id in gpt2_token_ids.items():
    decoded = decoder_tokenizer.decode([token_id])
    print(f"  {word} -> token_id: {token_id}, decoded: '{decoded}'")

gpt2_config = AutoConfig.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 加载JpegLM池化encoder
jpeglm_encoder = create_jpeglm_encoder_with_pooling("/root/autodl-tmp/MLLM/models/jpeg-lm", pooling_strategy='last').to(device)

# 分类投影层（将池化特征投影到gpt2的hidden_size）
proj = torch.nn.Linear(jpeglm_encoder.config.hidden_size, gpt2_config.n_embd).to(device)

# PrefixLM模型封装
class PrefixLMForClassification(torch.nn.Module):

    def __init__(self, encoder, proj, decoder, decoder_tokenizer, bos_token_id):
        super().__init__()
        self.encoder = encoder
        self.proj = proj
        self.decoder = decoder
        self.decoder_tokenizer = decoder_tokenizer
        self.bos_token_id = bos_token_id

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # 兼容transformers/peft的from_pretrained方法
        # 这里只是示例，实际可根据需要补充参数
        raise NotImplementedError("请用自定义初始化PrefixLMForClassification")

    def get_input_embeddings(self):
        # 兼容PEFT/transformers的get_input_embeddings方法
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 兼容PEFT/transformers的set_input_embeddings方法
        self.decoder.set_input_embeddings(value)

    @property
    def base_model(self):
        # 兼容PEFT的base_model属性
        return self

    @property
    def model(self):
        # 兼容PEFT的model属性
        return self

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # 兼容PEFT/transformers的prepare_inputs_for_generation方法
        # 直接委托给decoder处理
        return self.decoder.prepare_inputs_for_generation(input_ids, **kwargs)

    def resize_token_embeddings(self, new_num_tokens):
        # 兼容transformers的resize_token_embeddings方法
        return self.decoder.resize_token_embeddings(new_num_tokens)

    def tie_weights(self):
        # 兼容transformers的tie_weights方法
        if hasattr(self.decoder, 'tie_weights'):
            self.decoder.tie_weights()

    def can_generate(self):
        # 兼容transformers的can_generate方法
        return hasattr(self.decoder, 'generate')

    @property
    def config(self):
        # 兼容transformers的config属性
        return self.decoder.config

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size(0)
        # 允许encoder参数梯度传播
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoder_outputs.last_hidden_state
        prefix_embeds = self.proj(pooled).unsqueeze(1)  # [batch, 1, hidden]
        
        # 构建decoder输入：[prefix] 
        # 让模型在prefix位置直接预测分类token（不需要BOS）
        inputs_embeds = prefix_embeds  # [batch, 1, hidden]
        
        # 损失计算：让模型直接预测分类token
        if labels is not None:
            # labels形状应该与sequence长度匹配
            labels_for_loss = labels.unsqueeze(1)  # [batch, 1]
            outputs = self.decoder(inputs_embeds=inputs_embeds, labels=labels_for_loss)
        else:
            outputs = self.decoder(inputs_embeds=inputs_embeds)
        
        return outputs

    def generate(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = encoder_outputs.last_hidden_state
        prefix_embeds = self.proj(pooled).unsqueeze(1)  # [batch, 1, hidden]
        inputs_embeds = prefix_embeds
        outputs = self.decoder(inputs_embeds=inputs_embeds)
        logits = outputs.logits[:, 0, :]  # 取第一个（也是唯一的）位置的logits
        pred = torch.argmax(logits, dim=-1)
        return pred

# 数据集
class MNISTJpegBytesPrefixDataset(Dataset):
    def __init__(self, hf_dataset, encoder_tokenizer, decoder_tokenizer, digit_to_text, max_length=1024, image_size=28):
        self.dataset = hf_dataset
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.digit_to_text = digit_to_text
        self.max_length = max_length
        self.transform = create_preprocess_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = item['label']
        caption = self.digit_to_text[label]
        img = item['image'].convert("RGB")
        img = self.transform(img)
        jpeg_str = convert_img_to_bytes(img)
        input_ids = [self.encoder_tokenizer.bos_token_id] + self.encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
        input_ids = input_ids[:self.max_length]
        # 使用GPT2 tokenizer中的真实token ID
        caption = self.digit_to_text[label]
        label_token_id = gpt2_token_ids[caption]
        return {"input_ids": torch.tensor(input_ids), "label_id": torch.tensor(label_token_id)}

# 加载MNIST数据集
mnist_dataset = load_dataset("ylecun/mnist")
train_data = mnist_dataset["train"].select(range(min(args.num_train_samples, len(mnist_dataset["train"]))))
test_data = mnist_dataset["test"].select(range(min(args.num_eval_samples, len(mnist_dataset["test"]))))

train_dataset = MNISTJpegBytesPrefixDataset(train_data, encoder_tokenizer, decoder_tokenizer, digit_to_text, args.max_length, args.image_size)
val_dataset = MNISTJpegBytesPrefixDataset(test_data, encoder_tokenizer, decoder_tokenizer, digit_to_text, args.max_length, args.image_size)

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    label_ids = [item["label_id"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=encoder_tokenizer.pad_token_id)
    attention_mask = (input_ids_padded != encoder_tokenizer.pad_token_id).long()
    label_ids = torch.stack(label_ids)
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask, "labels": label_ids}

# MySeq2SeqTrainer集成
from hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments
bos_token_id = decoder_tokenizer.bos_token_id or decoder_tokenizer.eos_token_id or 50256
model = PrefixLMForClassification(jpeglm_encoder, proj, gpt2_model, decoder_tokenizer, bos_token_id)

def compute_metrics(pred):
    labels = pred.label_ids  # [batch_size] - 包含GPT2 token IDs
    preds = pred.predictions # [batch_size, seq_len, vocab_size] 或 [batch_size, seq_len]
    
    # 处理预测结果：如果是logits，取argmax；如果已经是token ids，直接使用
    if len(preds.shape) == 3:  # logits: [batch_size, seq_len, vocab_size]
        pred_token_ids = np.argmax(preds, axis=-1)  # [batch_size, seq_len]
    else:  # token ids: [batch_size, seq_len]
        pred_token_ids = preds
    
    # 取第1个位置的预测（现在序列长度为1）
    if pred_token_ids.shape[1] > 0:
        pred_tokens = pred_token_ids[:, 0]  # [batch_size]
    else:
        pred_tokens = np.zeros_like(labels)  # fallback
    
    # 计算准确率 - 比较GPT2 token IDs
    correct = (pred_tokens == labels).sum()
    total = len(labels)
    accuracy = correct / total if total > 0 else 0.0
    
    # 调试信息：打印前几个预测和标签
    if len(pred_tokens) > 0:
        print(f"[DEBUG] 前3个预测token IDs: {pred_tokens[:3]}")
        print(f"[DEBUG] 前3个真实token IDs: {labels[:3]}")
        for i in range(min(3, len(pred_tokens))):
            pred_word = decoder_tokenizer.decode([pred_tokens[i]])
            label_word = decoder_tokenizer.decode([labels[i]])
            print(f"[DEBUG] 样本{i}: 预测='{pred_word}' (id:{pred_tokens[i]}), 真实='{label_word}' (id:{labels[i]})")
    
    return {"accuracy": round(float(accuracy), 4)}

my_args = MySeq2SeqTrainingArguments(
    output_dir=args.output_dir,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    save_total_limit=args.save_total_limit,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    eval_strategy=args.eval_strategy,
    warmup_steps=args.warmup_steps,
    lr_scheduler_type=args.lr_scheduler_type,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    report_to=args.report_to if args.report_to not in [None, "None"] else None,
    fp16=args.fp16,
    bf16=args.bf16,
)

# 只对encoder开启梯度检查点
if hasattr(model.encoder, 'gradient_checkpointing_enable'):
    model.encoder.gradient_checkpointing_enable()

# LoRA配置

# 获取GPT2所有关键模块名称，作为modules_to_save
gpt2_modules = [
    "decoder.transformer.wte",      # word token embeddings
    "decoder.transformer.wpe",      # position embeddings
    "decoder.transformer.ln_f",     # final layer norm
    "decoder.lm_head",              # language modeling head
]

# 添加所有transformer层
for i in range(model.decoder.config.n_layer):
    gpt2_modules.extend([
        f"decoder.transformer.h.{i}.ln_1",
        f"decoder.transformer.h.{i}.attn.c_attn",
        f"decoder.transformer.h.{i}.attn.c_proj",
        f"decoder.transformer.h.{i}.ln_2", 
        f"decoder.transformer.h.{i}.mlp.c_fc",
        f"decoder.transformer.h.{i}.mlp.c_proj",
    ])

# 添加其他需要保存的模块
modules_to_save = gpt2_modules + [
    "proj",  # 投影层
]

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    target_modules=[
        # Encoder中的线性层 (JpegLM相关)
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention层
        "gate_proj", "up_proj", "down_proj",     # MLP层  
    ],
    # modules_to_save=modules_to_save,  # 保存GPT2全部模块和投影层
    lora_dropout=0.1,
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 打印LoRA训练参数统计
print("\n==== LoRA训练参数统计 ====")
model.print_trainable_parameters()

# 详细打印各层参数状态
print("\n==== 各层参数 requires_grad 状态 ====")
trainable_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"{name:80} requires_grad={param.requires_grad} ({param.numel():,} params)")

print(f"\n总训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
print("==== END ====")

trainer = MySeq2SeqTrainer(
    model=model,
    args=my_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=decoder_tokenizer,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

trainer.train()
trainer.save_model()
print(f"✓ 训练完成，模型已保存到 {args.output_dir}")
