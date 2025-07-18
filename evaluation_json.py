import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import PeftModel, AutoPeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Configuration ===
BASE_MODEL_NAME    = "/root/autodl-fs/models/Qwen3-8B"
PEFT_MODEL_PATH    = "/root/autodl-tmp/MLLM/trained_models/Qwen3-8B-lora-mnist10_jpeg_GRAD"
DATASET_PATH       = "/root/autodl-tmp/MLLM/datasets/mnist/mnist_jpeg_small_test.jsonl"
SPLIT              = "train"  # or 'test' / 'validation'
MAX_LENGTH         = 600
BATCH_SIZE         = 8
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PEFT_ENABLED       = True
LOAD_FROM_PEFT     = True
NUM_LABELS         = 10

# === Metrics ===

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

# === Load tokenizer and model ===
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

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


if LOAD_FROM_PEFT:
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        PEFT_MODEL_PATH,
        num_labels=NUM_LABELS,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
model.to(DEVICE)
model.eval()

# ensure pad token
pid_id = tokenizer.eos_token_id
tokenizer.pad_token_id = pid_id
model.config.pad_token_id = pid_id

# === Load and preprocess dataset ===
print(f"Loading dataset from {DATASET_PATH}...")
ds = load_dataset('json', data_files=DATASET_PATH, split=SPLIT)
# Use same split as trained: if train/test split done, adjust here

dataset = ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=[c for c in ds.column_names if c not in ('label', 'hex')]
)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# === Evaluation with Trainer ===
print("Starting evaluation...")
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./eval_out',
        per_device_eval_batch_size=BATCH_SIZE,
        do_train=False,
        do_predict=False,
        do_eval=True,
        bf16=True
    ),
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)
# Use evaluate to get accuracy, precision, recall, f1
eval_metrics = trainer.evaluate(eval_dataset=dataset)
print("Evaluation metrics:")
for k, v in eval_metrics.items():
    print(f"  {k}: {v}")
