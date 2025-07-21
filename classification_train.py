import os
import random
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, AutoPeftModelForSequenceClassification
from utils import rename_column, rename_column_byte_input, rename_column_hex

# ========================
# Configuration
# ========================
BASE_MODEL_NAME    = "/root/autodl-fs/models/Qwen3-8B"
PEFT_MODEL_PATH    = "/root/autodl-tmp/MLLM/trained_models/Qwen3-8B-lora-mnist10_jpeg_GRAD"
DATASET_PATH       = "/root/autodl-tmp/MLLM/datasets/mnist/mnist_jpeg.jsonl"
MAX_LENGTH         = 600
BATCH_SIZE         = 2
GRAD_ACC_STEPS     = 16
NUM_TRAIN_EPOCHS   = 3
NUM_LABELS         = 10
EVAL_ACCUMULATION_STEPS = 10
PEFT_ENABLED       = True
LOAD_FROM_PEFT     = True
REPORT_TO_WANDB    = True

# marker_id must match the integer inserted into bytes_annotated
MARKER_ID          = 256
MARKER_TOKEN       = " "

# ========================
# Load and preprocess dataset
# ========================
ds = load_dataset("json",
    data_files=DATASET_PATH,
    split="train"
)

ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds = ds["train"]
val_ds   = ds["test"]

# ========================
# Inspect a random sample
# ========================
example = train_ds.select([0])[0]
print("Sample example:", example)

# ========================
# Load tokenizer and register patch‐boundary token
# ========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# ========================
# Load model 
# ========================
if LOAD_FROM_PEFT:
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        PEFT_MODEL_PATH,
        num_labels=NUM_LABELS,
        is_trainable=True,
        device_map="auto",
        torch_dtype="auto"
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, 
        num_labels=NUM_LABELS, 
        device_map="auto", 
        torch_dtype="auto"
        )
    if PEFT_ENABLED:
        lora_cfg = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_cfg)

# ensure pad token
pid_id = tokenizer.eos_token_id
tokenizer.pad_token_id = pid_id
model.config.pad_token_id = pid_id

# ========================
# Tokenization function
# ========================
def tokenize_fn(batch):
    input_ids = []
    attention_mask = []
    for ba in batch["byte_array"]:
        # ba 中已包含 0–255 的字节和 256 的 marker_id，直接当作 input_id
        ids = list(ba)
        # 截断或填充到 MAX_LENGTH
        if len(ids) > MAX_LENGTH:
            ids = ids[:MAX_LENGTH]
            mask = [1] * MAX_LENGTH
        else:
            mask = [1] * len(ids) + [0] * (MAX_LENGTH - len(ids))
            ids = ids + [pid_id] * (MAX_LENGTH - len(ids))      
        input_ids.append(ids)
        attention_mask.append(mask)
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
    }

columns_to_remove = [c for c in train_ds.column_names
                     if c not in ("label",)]

# Apply to datasets
train_ds = train_ds.rename_column("label", "labels")
val_ds   = val_ds.rename_column("label", "labels")
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=columns_to_remove)
val_ds   = val_ds.map(  tokenize_fn, batched=True, remove_columns=columns_to_remove)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Inspect tokens
print("Tokens for sample:", tokenizer.convert_ids_to_tokens(train_ds[0]["input_ids"]))

# ========================
# Metrics
# ========================
def compute_metrics(pred):
    logits, labels = pred
    if isinstance(logits, tuple): logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

data_collator = DataCollatorWithPadding(tokenizer)

# ========================
# Training arguments
# ========================
training_args = TrainingArguments(
    output_dir=f"autodl-tmp/MLLM/checkpoints/{os.path.basename(BASE_MODEL_NAME)}-lora",
    run_name=f"{os.path.basename(BASE_MODEL_NAME)}-lora-{os.path.basename(DATASET_PATH)}",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=2e-4 if PEFT_ENABLED else 1e-5,
    # warmup_steps=200, 
    report_to="wandb" if REPORT_TO_WANDB else "none",
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    bf16=True,    
)

# ========================
# Trainer
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# ========================
# Prediction helper
# ========================
def predict_batch(texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    for k,v in enc.items():
        enc[k] = v.to(device)
    with torch.no_grad():
        out = model(**enc)
        probs = out.logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
    return preds.cpu().tolist(), probs.cpu().tolist()

# Test prediction
sample_texts = [train_ds[random.randrange(len(train_ds))]["input_ids"]]
# (you may convert back to tokens or text as needed)
preds, probs = predict_batch(sample_texts)
print("Prediction:", preds, "Confidences:", probs)
