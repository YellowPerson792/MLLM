import os
import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from transformers import AutoTokenizer, GPT2Config, default_data_collator

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

os.environ["WANDB_DISABLED"] = "true"
class config:
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    NUM_WORKERS = mp.cpu_count()
    EPOCHS = 3
    IMG_SIZE = (224,224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95

def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token

import evaluate
rouge = evaluate.load("rouge")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    pred_str = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True, clean_up_tokenization_spaces=True)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    res = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"], use_stemmer=True)
    return {"rouge2_fmeasure": round(res["rouge2"] * 100, 4)}

from datasets import load_dataset
transforms = T.Compose([
    T.Resize(config.IMG_SIZE),
    T.ToTensor(),
    # T.Normalize(
    #     mean=0.5,
    #     std=0.5
    # )
])
ds = load_dataset("jxie/flickr8k")
train_ds = ds["train"]
val_ds = ds["validation"]
val_ds = val_ds.select(range(100))
test_ds = ds["test"] if "test" in ds else None
print(train_ds[0])

class ImgDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, feature_extractor, transform=None, max_length=50):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]
        caption = item.get("text", item.get("caption_0", ""))
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

train_dataset = ImgDataset(train_ds, tokenizer=tokenizer, feature_extractor=feature_extractor, transform=transforms)
val_dataset = ImgDataset(val_ds, tokenizer=tokenizer, feature_extractor=feature_extractor, transform=transforms)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
model.config.use_cache = False

training_args = Seq2SeqTrainingArguments(
    output_dir='VIT_large_gpt2',
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    eval_strategy="epoch",
    eval_steps=10,
    do_train=True,
    do_eval=True,
    logging_steps=128,
    save_steps=2048,
    warmup_steps=1024,
    learning_rate=5e-5,
    num_train_epochs=config.EPOCHS,
    overwrite_output_dir=True,
    save_total_limit=1,
)

trainer = Seq2SeqTrainer(
    tokenizer=feature_extractor,
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)
trainer.train()

trainer.save_model('VIT_large_gpt2')

img = Image.open("/kaggle/input/flickr8k/Images/1001773457_577c3a7d70.jpg").convert("RGB")
print(img)
generated_caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cuda"))[0])
print('\033[96m' + generated_caption[:85] + '\033[0m')

img = Image.open("/kaggle/input/flickr8k/Images/1000268201_693b08cb0e.jpg").convert("RGB")
print(img)
generated_caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cuda"))[0])
print('\033[96m' + generated_caption[:120] + '\033[0m')
