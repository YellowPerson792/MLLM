#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 评估脚本：加载ViT-GPT2检查点并在验证集上评测ROUGE2
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm

import glob

class config:
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    MAX_LEN = 128
    IMG_SIZE = (224, 224)
    LABEL_MASK = -100
    CHECKPOINT_GLOB = "/root/autodl-tmp/MLLM/ImageCaption/VIT_large_gpt2/checkpoint-*"  # checkpoint通配符

# 自动查找最新checkpoint目录
def get_latest_checkpoint(ckpt_glob):
    candidates = glob.glob(ckpt_glob)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for pattern: {ckpt_glob}")
    # 取编号最大的那个checkpoint
    candidates = sorted(candidates, key=lambda x: int(x.split("-")[-1]))
    return candidates[-1]

# 加载特征提取器和分词器
feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token

# 加载模型
ckpt_path = get_latest_checkpoint(config.CHECKPOINT_GLOB)
print(f"Using checkpoint: {ckpt_path}")
model = VisionEncoderDecoderModel.from_pretrained(ckpt_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

rouge = evaluate.load("rouge")
def compute_metrics(preds, labels):
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    res = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"], use_stemmer=True)
    return {"rouge2_fmeasure": round(res["rouge2"] * 100, 4)}

transforms = T.Compose([
    T.Resize(config.IMG_SIZE),
    T.ToTensor(),
])

import pandas as pd
from sklearn.model_selection import train_test_split

# 加载Kaggle csv数据集
csv_path = "/root/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/captions.txt"
img_root = "/root/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/Images"
df = pd.read_csv(csv_path)
_, val_df = train_test_split(df, test_size=0.2, random_state=42)
val_df = val_df.head(100)


# 兼容Kaggle csv的Dataset
class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, transform=None, max_length=50):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.max_length = max_length
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        captions = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        ).input_ids
        captions = [token if token != self.tokenizer.pad_token_id else -100 for token in captions]
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(captions)
        }
        return encoding



val_dataset = ImgDataset(val_df, root_dir=img_root, tokenizer=tokenizer, feature_extractor=feature_extractor, transform=transforms, max_length=config.MAX_LEN)

# 设置评估batch_size
EVAL_BATCH_SIZE = 4
from torch.utils.data import DataLoader
val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)



# 推理与评测（批量）
all_preds = []
all_labels = []
print(f"Running inference on validation set with batch_size={EVAL_BATCH_SIZE}...")
for batch in tqdm(val_loader, desc="Evaluating", ncols=80):
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"]
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=config.MAX_LEN)
    all_preds.append(generated_ids.cpu().numpy())
    all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
metrics = compute_metrics(all_preds, all_labels)
print("Evaluation metrics:", metrics)

# 生成部分样例
print("\nSample predictions:")
for i in range(3):
    row = val_df.iloc[i]
    img_path = os.path.join(img_root, row.image)
    img = Image.open(img_path).convert("RGB")
    pixel_values = feature_extractor(transforms(img), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output = model.generate(pixel_values, max_length=config.MAX_LEN)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Sample {i+1} caption:", caption)
