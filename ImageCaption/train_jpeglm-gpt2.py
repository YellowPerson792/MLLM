# 示例运行命令：
# python /root/autodl-tmp/MLLM/ImageCaption/train_jpeglm-gpt2.py --train_batch_size 2 --eval_batch_size 2 --eval_strategy steps --eval_steps 128 --logging_steps 128 --save_steps 512 --warmup_steps 1024 --learning_rate 5e-5 --num_train_epochs 3 --save_total_limit 1 --lr_scheduler_type linear --gradient_accumulation_steps 8 --report_to None --fp16 --max_length 2048 ----image_size 224

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
from jpeglm.models.jpeglm_encoder import create_jpeglm_encoder
from sklearn.model_selection import train_test_split
import datasets
import multiprocessing as mp
from utils.data_utils import convert_img_to_bytes, create_preprocess_transform
from peft import get_peft_model, LoraConfig, TaskType

# 配置
class config:
    ENCODER = "/root/autodl-fs/models/jpeg-lm"
    DECODER = "gpt2"
    SEED = 42
    MAX_LEN = 18
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
parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/MLLM/checkpoints/jpeglm-gpt2-captioning")
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
parser.add_argument('--image_size', type=int, default=224, help='输入图片resize的边长')
parser.add_argument('--bit_flip_prob', type=float, default=0.0, help='JPEG比特流随机翻转概率')
parser.add_argument('--max_length', type=int, default=2000, help='JPEG比特流token序列最大长度')
args = parser.parse_args()


# 数据集
csv_path = "/root/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/captions.txt"
img_root = "/root/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/Images"
df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
val_df = val_df.head(100)

class JpegBytesDataset(Dataset):
    def __init__(self, df, root_dir, encoder_tokenizer, decoder_tokenizer, max_length=2000, image_size=224, bit_flip_prob=0.0):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_length = max_length
        self.transform = create_preprocess_transform(image_size)
        self.bit_flip_prob = bit_flip_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        jpeg_str = convert_img_to_bytes(img, bit_flip_prob=self.bit_flip_prob)
        input_ids = [self.encoder_tokenizer.bos_token_id] + self.encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
        input_ids = input_ids[:self.max_length] + [self.encoder_tokenizer.pad_token_id] * max(0, self.max_length - len(input_ids))
        labels = self.decoder_tokenizer(
            caption,
            padding='max_length',
            max_length=50,
            truncation=True
        ).input_ids
        labels = [token if token != self.decoder_tokenizer.pad_token_id else -100 for token in labels]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}

train_dataset = JpegBytesDataset(
    train_df, img_root, encoder_tokenizer, decoder_tokenizer,
    image_size=args.image_size,
    bit_flip_prob=args.bit_flip_prob,
    max_length=args.max_length
)
val_dataset = JpegBytesDataset(
    val_df, img_root, encoder_tokenizer, decoder_tokenizer,
    image_size=args.image_size,
    bit_flip_prob=args.bit_flip_prob,
    max_length=args.max_length
)


# 构建EncoderDecoderModel，使用ViTEncoderWrapper

gpt2_config = GPT2Config.from_pretrained(config.DECODER)
gpt2_config.add_cross_attention = True
gpt2 = GPT2LMHeadModel.from_pretrained(config.DECODER, config=gpt2_config)
# 用JpegLMEncoder作为encoder
encoder = create_jpeglm_encoder(config.ENCODER)
model = EncoderDecoderModel(encoder=encoder, decoder=gpt2)
# 只设置结构/训练相关参数
model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
model.config.pad_token_id = decoder_tokenizer.pad_token_id
model.config.eos_token_id = decoder_tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.main_input_name = "input_ids"

# 使用新版transformers的GenerationConfig
generation_config = GenerationConfig(
    max_length=18,
    num_beams=1,
    no_repeat_ngram_size=3,
    decoder_start_token_id=model.config.decoder_start_token_id,
    bos_token_id=decoder_tokenizer.bos_token_id,
    pad_token_id=decoder_tokenizer.pad_token_id,
    eos_token_id=decoder_tokenizer.eos_token_id,
)
model.generation_config = generation_config

# 评测指标
import nltk, evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

rouge = evaluate.load("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = decoder_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # 尝试转换为torch.Tensor并替换-100
    if not isinstance(labels_ids, torch.Tensor):
        labels_ids = torch.tensor(labels_ids)
    labels_ids = labels_ids.clone()
    labels_ids[labels_ids == -100] = decoder_tokenizer.pad_token_id
    label_str = decoder_tokenizer.batch_decode(labels_ids.tolist(), skip_special_tokens=True)
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
        "rouge2_fmeasure": round(rouge_output, 4),
        "bleu1": round(np.mean(bleu_1_scores), 4),
        "bleu4": round(np.mean(bleu_4_scores), 4),
    }

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

trainer = MySeq2SeqTrainer(
    model=model,
    args=my_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=decoder_tokenizer,
    compute_metrics=compute_metrics
)

from peft import get_peft_model, LoraConfig, TaskType


# 先开启梯度检查点
model.gradient_checkpointing_enable()
model.to(device)

# LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # 可调
    lora_alpha=32,  # 可调
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # encoder/decoder常见
        "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj", "c_fc", "q_attn"
    ],
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainer.train()
trainer.save_model()


# 生成样例
def generate_caption(image_path):
    img = Image.open(image_path).convert("RGB")
    img = create_preprocess_transform(args.image_size)(img)
    jpeg_str = convert_img_to_bytes(img, bit_flip_prob=args.bit_flip_prob)
    input_ids = [encoder_tokenizer.bos_token_id] + encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
    input_ids = input_ids[:args.max_length] + [encoder_tokenizer.pad_token_id] * max(0, args.max_length - len(input_ids))
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    generated_ids = model.generate(
        inputs=input_ids,
        generation_config=generation_config,
    )
    generated_caption = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_caption

for idx in range(3):
    image_path = os.path.join(img_root, val_df.iloc[idx].image)
    generated_caption = generate_caption(image_path)
    real_caption = val_df.iloc[idx].caption
    print(f"Image {idx + 1}:")
    print("Real Caption:", real_caption)
    print("Generated Caption:", generated_caption)
    img = Image.open(image_path)
    # 保存到output子目录
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_img_path = os.path.join(output_dir, f"output_image_{idx+1}.jpg")
    img.save(save_img_path)
    # 保存caption文本
    caption_path = os.path.join(output_dir, f"output_caption_{idx+1}.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(f"Real Caption: {real_caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")



