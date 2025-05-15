from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Qwen3ForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch 
import numpy as np
from transformers import DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig, AutoPeftModelForSequenceClassification
from utils import *
import os


BASE_MODEL_NAME = "/root/autodl-tmp/MLLM/models_cache/Qwen/Qwen3-8B"
MAX_LENGTH = 1024
BATCH_SIZE = 1
NUM_TRAIN_EPOCHS = 4
NUM_LABELS = 10
EVAL_ACCUMULATION_STEPS = 10

byte_spilt = 1
hex_input = 0
peft_en = 1
load_from_peft = 1
report_en = 1

torch.set_printoptions(threshold=float('inf'))

dataset = load_dataset("json", data_files="/root/autodl-tmp/MLLM/datasets/cifar10_jpeg.jsonl", split="train")
dataset = dataset.map(rename_column_hex) if hex_input \
    else dataset.map(rename_column_byte_input) if byte_spilt else dataset.map(rename_column)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]
dataset_example = train_ds.select(range(1))
sample_text = []
for example in dataset_example:
    sample_text.append(example['text'])
    # print(example)      # 查看数据集

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME, 
#     num_labels=NUM_LABELS, 
#     device_map="auto", 
#     torch_dtype="auto"
# )

model = AutoPeftModelForSequenceClassification.from_pretrained(
    "/root/autodl-tmp/MLLM/models_cache/Qwen/Qwen3-8B-lora",
    num_labels=NUM_LABELS, 
    is_trainable=True,
    device_map="auto", 
    torch_dtype="auto"
)

# 修复 padding 问题
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        is_split_into_words=True if byte_spilt else False,
        padding="longest",
        truncation=True,
        max_length=MAX_LENGTH
    )


train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# print(tokenizer.convert_ids_to_tokens(train_ds[0]['input_ids']))        # 查看token
# print(train_ds[0])      # 查看模型输入

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]
                
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=f"autodl-tmp/MLLM/checkpoints/{os.path.basename(BASE_MODEL_NAME)}-lora",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=1e-5 if peft_en else 5e-5,
    report_to="wandb" if report_en else "none",
    eval_strategy="epoch",
    logging_strategy="steps", 
    logging_steps=500,
    logging_dir="./logs",
    save_strategy="steps",
    # eval_steps=5000,
    save_steps=5000,
    # load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="accuracy",  
    greater_is_better=True,             
    bf16=True,       
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    label_names=['lable_name']
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks=[NStepsCallback(save_every_steps=20, output_dir="/root/autodl-tmp/MLLM/trained_models")],
)

qwen3_lora_config = LoraConfig(
    r=64,  # LoRA rank，常用 8/16/32/64，越大效果越好，但显存越高
    lora_alpha=128,  # 一般设置为 2 * r
    target_modules=["q_proj", "v_proj"],  # 只插入到注意力机制里，节省资源
    lora_dropout=0.05,
    bias="none",  # 不修改原始 bias
    task_type=TaskType.SEQ_CLS,
)

if(peft_en):
    if not load_from_peft:
        model = get_peft_model(model, qwen3_lora_config)
    model.print_trainable_parameters()  

print("!!!!!")
print(isinstance(model, PeftModel))
print("~~~~~")

trainer.train()

def predict_batch(texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # texts 应该是一个字符串列表
    inputs = tokenizer(
        texts,
        is_split_into_words=True if byte_spilt else False,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    # 将所有输入移到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)  # logits shape: [batch_size, num_classes]
        probs = outputs.logits.softmax(dim=-1)  # shape: [batch_size, num_classes]
        preds = probs.argmax(dim=-1)            # shape: [batch_size]

    # 转成 Python 数据类型
    preds = preds.tolist()         
    probs = probs.tolist()          

    return preds, probs


preds, probs = predict_batch(sample_text)

for cls, conf in zip(preds, probs):
    print("预测类别:", cls, "置信度:", conf)


