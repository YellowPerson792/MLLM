# 示例运行命令：
# python /root/autodl-tmp/MLLM/ImageCaption/train_jpeglm-gpt2_cls.py --train_batch_size 2 --eval_batch_size 2 --eval_strategy steps --eval_steps 128 --logging_steps 64 --save_steps 512 --warmup_steps 512 --learning_rate 2e-4 --num_train_epochs 3 --save_total_limit 6 --lr_scheduler_type linear --gradient_accumulation_steps 8 --report_to wandb --bf16 --max_length 1024 --image_size 96 --num_train_samples 6000 --num_eval_samples 16

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_style_trainer import MySeq2SeqTrainingArguments
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import EncoderDecoderModel, GPT2LMHeadModel, ViTFeatureExtractor, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator, GenerationConfig, GPT2Config
from torch.nn.utils.rnn import pad_sequence
from jpeglm.models.jpeglm_encoder import JpegLMEncoderDecoderModelWithPooling, create_jpeglm_encoder, create_jpeglm_encoder_with_pooling
from sklearn.model_selection import train_test_split
import datasets
import multiprocessing as mp
from utils.data_utils import convert_img_to_bytes, create_preprocess_transform
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# 配置
class config:
    ENCODER = "/root/autodl-tmp/MLLM/models/jpeg-lm"
    DECODER = "gpt2"
    SEED = 42
    MAX_LEN = 1
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    NUM_WORKERS = mp.cpu_count()
    IMG_SIZE = (224, 224)
    TOP_K = 1000
    TOP_P = 0.95
os.environ["WANDB_DISABLED"] = "true"
script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 分别为encoder和decoder准备tokenizer
encoder_tokenizer = AutoTokenizer.from_pretrained(config.ENCODER)  # 用于JPEG比特流tokenize
decoder_tokenizer = AutoTokenizer.from_pretrained(config.DECODER)  # 用于caption tokenize
decoder_tokenizer.pad_token = decoder_tokenizer.unk_token

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/MLLM/checkpoints/jpeglm-gpt2-mnist-classification")
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--eval_strategy', type=str, default="epoch")
parser.add_argument('--eval_steps', type=int, default=128)
parser.add_argument('--logging_steps', type=int, default=128)
parser.add_argument('--save_steps', type=int, default=128)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--save_total_limit', type=int, default=1)
parser.add_argument('--lr_scheduler_type', type=str, default="linear")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--report_to', type=str, default=None)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--image_size', type=int, default=28, help='输入图片resize的边长')
parser.add_argument('--bit_flip_prob', type=float, default=0.0, help='JPEG比特流随机翻转概率')
parser.add_argument('--max_length', type=int, default=1024, help='JPEG比特流token序列最大长度')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='从指定checkpoint恢复训练状态（不是预训练权重）')
parser.add_argument('--num_train_samples', type=int, default=6000, help='用于训练的样本数')
parser.add_argument('--num_eval_samples', type=int, default=1000, help='用于评估的样本数')
args = parser.parse_args()


# 数据集 - 使用MNIST数据集
from datasets import load_dataset

# 加载MNIST数据集
print("正在加载MNIST数据集...")
mnist_dataset = load_dataset("ylecun/mnist")
train_data = mnist_dataset["train"]
test_data = mnist_dataset["test"]


# 按参数裁剪数据集
train_data = train_data.select(range(min(args.num_train_samples, len(train_data))))
test_data = test_data.select(range(min(args.num_eval_samples, len(test_data))))
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

# MNIST数字标签映射到文本
digit_to_text = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

# ====== TinyDecoder词表和模块 ======
tiny_vocab = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
word2id = {w: i for i, w in enumerate(tiny_vocab)}
id2word = {i: w for i, w in enumerate(tiny_vocab)}

import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class TinyDecoderConfig(PretrainedConfig):
    def __init__(self, vocab_size=10, hidden_size=32, num_layers=1, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

class TinyTransformerDecoder(PreTrainedModel):
    config_class = TinyDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 2,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, encoder_hidden_states, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        tgt_mask = None
        memory = encoder_hidden_states
        out = self.decoder(x, memory, tgt_mask=tgt_mask)
        logits = self.lm_head(out)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}

class MNISTJpegBytesDataset(Dataset):
    def __init__(self, hf_dataset, encoder_tokenizer, decoder_tokenizer, digit_to_text, max_length=1024, image_size=28, bit_flip_prob=0.0):
        self.dataset = hf_dataset
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.digit_to_text = digit_to_text
        self.max_length = max_length
        self.transform = create_preprocess_transform(image_size)
        self.bit_flip_prob = bit_flip_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # MNIST数据格式: {'image': PIL_image, 'label': int}
        label = item['label']
        caption = self.digit_to_text[label]  # 根据数字标签生成文本
        # 处理图像
        img = item['image'].convert("RGB")  # MNIST原本是灰度图，转为RGB
        img = self.transform(img)
        # 将图像转换为JPEG字节流
        jpeg_str = convert_img_to_bytes(img, bit_flip_prob=self.bit_flip_prob)
        input_ids = [self.encoder_tokenizer.bos_token_id] + self.encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
        input_ids = input_ids[:self.max_length]
        # 处理标签文本（只用tiny_vocab）
        label_id = word2id[caption]
        labels = torch.tensor([label_id])  # 只输出一个token
        return {"input_ids": torch.tensor(input_ids), "labels": labels}


train_dataset = MNISTJpegBytesDataset(
    train_data, encoder_tokenizer, decoder_tokenizer, digit_to_text,
    image_size=args.image_size,
    bit_flip_prob=args.bit_flip_prob,
    max_length=args.max_length
)
val_dataset = MNISTJpegBytesDataset(
    test_data, encoder_tokenizer, decoder_tokenizer, digit_to_text,
    image_size=args.image_size,
    bit_flip_prob=args.bit_flip_prob,
    max_length=args.max_length
)

# 动态padding的collate_fn
def dynamic_pad_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    # pad到本batch最大长度
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=encoder_tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    # attention_mask: 1 for non-pad, 0 for pad
    attention_mask = (input_ids_padded != encoder_tokenizer.pad_token_id).long()
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask, "labels": labels_padded}


# 构建EncoderDecoderModel，使用ViTEncoderWrapper


# 用JpegLMEncoder作为encoder
encoder = create_jpeglm_encoder_with_pooling(config.ENCODER, pooling_strategy='last')
# TinyDecoder
tiny_config = TinyDecoderConfig(vocab_size=len(tiny_vocab))
tiny_decoder = TinyTransformerDecoder(tiny_config)
model = JpegLMEncoderDecoderModelWithPooling(encoder=encoder, decoder=tiny_decoder)

# 设置结构/训练相关参数
model.config.decoder_start_token_id = 0  # 'zero'对应id=0
model.config.pad_token_id = -100  # 不使用pad
model.config.eos_token_id = None
model.config.vocab_size = len(tiny_vocab)
model.main_input_name = "input_ids"

# GenerationConfig（只生成1个token）
generation_config = GenerationConfig(
    max_new_tokens=1,
    num_beams=1,
    decoder_start_token_id=0,
    bos_token_id=0,
    pad_token_id=-100,
    eos_token_id=None,
)
model.generation_config = generation_config

# 评测指标
import nltk, evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

rouge = evaluate.load("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    # 动态pad到同一长度
    import torch
    from torch.nn.utils.rnn import pad_sequence
    # 转为list of tensor
    labels_ids = [torch.tensor(x) for x in labels_ids]
    pred_ids = [torch.tensor(x) for x in pred_ids]
    # pad
    labels_ids = pad_sequence(labels_ids, batch_first=True, padding_value=decoder_tokenizer.pad_token_id)
    pred_ids = pad_sequence(pred_ids, batch_first=True, padding_value=decoder_tokenizer.pad_token_id)
    # decode
    pred_str = decoder_tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
    labels_ids[labels_ids == -100] = decoder_tokenizer.pad_token_id
    label_str = decoder_tokenizer.batch_decode(labels_ids.tolist(), skip_special_tokens=True)
    
    # 计算分类准确率
    correct = 0
    total = 0
    for pred_text, label_text in zip(pred_str, label_str):
        # 从生成的文本中提取数字
        pred_digit = extract_digit_from_text(pred_text)
        label_digit = extract_digit_from_text(label_text)
        
        if pred_digit is not None and label_digit is not None:
            if pred_digit == label_digit:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # 计算传统的文本生成指标
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"]
    bleu_1_scores = []
    bleu_4_scores = []
    for ref, pred in zip(label_str, pred_str):
        reference = [nltk.word_tokenize(ref)]
        candidate = nltk.word_tokenize(pred)
        smoothing_function = SmoothingFunction().method4
        bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
        bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
        bleu_1_scores.append(bleu_1)
        bleu_4_scores.append(bleu_4)
    
    return {
        "accuracy": round(accuracy, 4),  # 新增分类准确率
        "rouge2_fmeasure": round(rouge_output, 4),
        "bleu1": round(np.mean(bleu_1_scores), 4),
        "bleu4": round(np.mean(bleu_4_scores), 4),
    }

def extract_digit_from_text(text):
    """从生成的文本中提取数字"""
    import re
    # 查找数字词
    digit_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    
    text = text.lower()
    for word, digit in digit_words.items():
        if word in text:
            return digit
    
    # 如果没有找到数字词，尝试查找数字
    digits = re.findall(r'\d', text)
    if digits:
        return int(digits[0])
    
    return None

# ====== 使用自定义训练框架 ======
from hf_style_trainer import MySeq2SeqTrainer

my_args = MySeq2SeqTrainingArguments(
    output_dir=args.output_dir,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    warmup_steps=args.warmup_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    save_total_limit=args.save_total_limit,
    lr_scheduler_type=args.lr_scheduler_type,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    report_to=args.report_to if args.report_to not in [None, "None"] else None,
    fp16=args.fp16,
    bf16=args.bf16
)

# ====== 只训练cross-attention层，其余参数全部冻结 ======
# for name, param in model.named_parameters():
#     # cross-attention层名一般包含"crossattention"或"cross_attention"
#     if ("crossattention" in name.lower()) or ("cross_attention" in name.lower()):
#         param.requires_grad = True
#     else:
#         param.requires_grad = False
# print("仅训练cross-attention层，其余参数已冻结。")


# 自动收集所有decoder.transformer.h的子模块名
h_modules = [f"decoder.transformer.h.{i}" for i in range(model.decoder.config.n_layer)]
modules_to_save = h_modules + [
    "decoder.transformer.ln_f",
    "decoder.lm_head",
    "enc_to_dec_proj"
]
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # 可调
    lora_alpha=32,  # 可调
    target_modules=[
        # 通用的attention和MLP模块名，应该能匹配到encoder中的模块
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention层
        "gate_proj", "up_proj", "down_proj",     # MLP层
        "fc1", "fc2", "dense"                    # 其他可能的线性层
    ],
    modules_to_save=modules_to_save,
    lora_dropout=0.1
)

# 开启梯度检查点
model.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)


# 打印所有参数的requires_grad状态
print("\n==== 各层参数 requires_grad 状态 ====")
for name, param in model.named_parameters():
    print(f"{name:80} requires_grad={param.requires_grad}")
model.print_trainable_parameters()
print("==== END ====")

trainer = MySeq2SeqTrainer(
    model=model,
    args=my_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=decoder_tokenizer,
    compute_metrics=compute_metrics,
    data_collator=dynamic_pad_collate_fn
)


trainer.train()
trainer.save_model()


# 生成样例
def generate_classification_from_image(pil_image):
    """根据PIL图像生成分类文本"""
    img = pil_image.convert("RGB")
    img = create_preprocess_transform(args.image_size)(img)
    jpeg_str = convert_img_to_bytes(img, bit_flip_prob=args.bit_flip_prob)
    input_ids = [encoder_tokenizer.bos_token_id] + encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
    input_ids = input_ids[:args.max_length] + [encoder_tokenizer.pad_token_id] * max(0, args.max_length - len(input_ids))
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    generated_ids = model.generate(
        inputs=input_ids,
        generation_config=generation_config,
    )
    generated_text = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

print("\n==== 生成分类样例 ====")
for idx in range(3):
    # 从测试集中获取样本
    sample = test_data[idx]
    pil_image = sample['image']
    true_label = sample['label']
    true_caption = digit_to_text[true_label]
    
    generated_caption = generate_classification_from_image(pil_image)
    
    print(f"Sample {idx + 1}:")
    print(f"True Label: {true_label}")
    print(f"True Caption: {true_caption}")
    print(f"Generated Caption: {generated_caption}")
    
    # 提取预测的数字
    pred_digit = extract_digit_from_text(generated_caption)
    print(f"Predicted Digit: {pred_digit}")
    print(f"Correct: {pred_digit == true_label}")
    print("-" * 50)
    
    # 保存到output子目录
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图像
    save_img_path = os.path.join(output_dir, f"mnist_sample_{idx+1}.png")
    pil_image.save(save_img_path)
    
    # 保存分类结果
    result_path = os.path.join(output_dir, f"classification_result_{idx+1}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"True Label: {true_label}\n")
        f.write(f"True Caption: {true_caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")
        f.write(f"Predicted Digit: {pred_digit}\n")
        f.write(f"Correct: {pred_digit == true_label}\n")

print("✓ 样例生成完成，结果保存在 output/ 目录中")