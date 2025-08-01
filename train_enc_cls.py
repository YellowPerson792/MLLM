# 示例运行命令：
# python /root/autodl-tmp/MLLM/train_enc_cls.py --train_batch_size 2 --eval_batch_size 2 --eval_strategy steps --eval_steps 512 --logging_steps 64 --save_steps 512 --warmup_steps 512 --learning_rate 2e-4 --num_train_epochs 3 --save_total_limit 6 --lr_scheduler_type linear --gradient_accumulation_steps 8 --report_to wandb --bf16 --max_length 1024 --image_size 96 --num_train_samples 6000 --num_eval_samples 16

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from jpeglm.models.jpeglm_encoder import create_jpeglm_encoder_cls_model
from utils.data_utils import convert_img_to_bytes, create_preprocess_transform
from peft import get_peft_model, LoraConfig, TaskType
import argparse
from datasets import load_dataset
import torch.nn.functional as F

# ====== 引入自定义Trainer和训练参数 ======
from ImageCaption.hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments

# 配置
class config:
    ENCODER = "/root/autodl-fs/models/jpeg-lm"
    NUM_CLASSES = 10

os.environ["WANDB_DISABLED"] = "true"
script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizer
encoder_tokenizer = AutoTokenizer.from_pretrained(config.ENCODER)


# 命令行参数与train_jpeglm-gpt2_cls.py保持一致
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/MLLM/checkpoints/jpeglm-enc-mnist-classification")
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

# 数据集
print("正在加载MNIST数据集...")
mnist_dataset = load_dataset("ylecun/mnist")
train_data = mnist_dataset["train"].select(range(min(args.num_train_samples, len(mnist_dataset["train"]))))
test_data = mnist_dataset["test"].select(range(min(args.num_eval_samples, len(mnist_dataset["test"]))))
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

class MNISTJpegBytesClsDataset(Dataset):
    def __init__(self, hf_dataset, encoder_tokenizer, max_length=1024, image_size=28, bit_flip_prob=0.0):
        self.dataset = hf_dataset
        self.encoder_tokenizer = encoder_tokenizer
        self.max_length = max_length
        self.transform = create_preprocess_transform(image_size)
        self.bit_flip_prob = bit_flip_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = item['label']
        img = item['image'].convert("RGB")
        img = self.transform(img)
        jpeg_str = convert_img_to_bytes(img, bit_flip_prob=self.bit_flip_prob)
        input_ids = [self.encoder_tokenizer.bos_token_id] + self.encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
        input_ids = input_ids[:self.max_length]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(label)}

train_dataset = MNISTJpegBytesClsDataset(
    train_data, encoder_tokenizer,
    image_size=args.image_size,
    bit_flip_prob=args.bit_flip_prob,
    max_length=args.max_length
)
val_dataset = MNISTJpegBytesClsDataset(
    test_data, encoder_tokenizer,
    image_size=args.image_size,
    bit_flip_prob=args.bit_flip_prob,
    max_length=args.max_length
)

def dynamic_pad_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=encoder_tokenizer.pad_token_id)
    attention_mask = (input_ids_padded != encoder_tokenizer.pad_token_id).long()
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask, "labels": labels}

# 构建JpegLMEncoderForClassification
model = create_jpeglm_encoder_cls_model(
    model_name_or_path=config.ENCODER,
    num_classes=config.NUM_CLASSES,
    pooling="mean"
)
model = model.to(device)

# LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2", "dense"
    ],
    modules_to_save=[
        "classifier"
    ],
    lora_dropout=0.1
)

model.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)

# 打印所有参数的requires_grad状态
print("\n==== 各层参数 requires_grad 状态 ====")
for name, param in model.named_parameters():
    print(f"{name:80} requires_grad={param.requires_grad}")
model.print_trainable_parameters()
print("==== END ====")

# 训练与评估
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Eval Accuracy: {acc:.4f}")
    return acc


# ====== 自定义训练参数 ======
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
    bf16=args.bf16,
)

# ====== 评测指标 ======
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    acc = (preds == labels).sum() / len(labels)
    return {"accuracy": round(float(acc), 4)}


# ====== 自定义分类Trainer，重写evaluate方法 ======
class ClsTrainer(MySeq2SeqTrainer):
    def evaluate(self, eval_dataset=None, desc="Eval", ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        self.model.eval()
        total, correct = 0, 0
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataset, desc=f"{desc} (custom)"):
                device = self.args.device if hasattr(self.args, 'device') else self.model.device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item() * labels.size(0)
        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        print(f"[Custom Eval] Loss: {avg_loss:.4f}  Accuracy: {acc:.4f}  (Total: {total})")
        return avg_loss, acc

trainer = ClsTrainer(
    model=model,
    args=my_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=dynamic_pad_collate_fn
)

trainer.train()
trainer.save_model()

# 生成样例
def predict_single_image(pil_image):
    model.eval()
    img = pil_image.convert("RGB")
    img = create_preprocess_transform(args.image_size)(img)
    jpeg_str = convert_img_to_bytes(img, bit_flip_prob=args.bit_flip_prob)
    input_ids = [encoder_tokenizer.bos_token_id] + encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
    input_ids = input_ids[:args.max_length]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = (input_ids != encoder_tokenizer.pad_token_id).long()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = logits.argmax(dim=-1).item()
    return pred

print("\n==== 生成分类样例 ====")
for idx in range(3):
    sample = test_data[idx]
    pil_image = sample['image']
    true_label = sample['label']
    pred_digit = predict_single_image(pil_image)
    print(f"Sample {idx+1}: True Label: {true_label}, Predicted: {pred_digit}, Correct: {pred_digit == true_label}")
    # 保存图片和结果
    output_dir = os.path.join(script_dir, "output_cls")
    os.makedirs(output_dir, exist_ok=True)
    pil_image.save(os.path.join(output_dir, f"mnist_sample_{idx+1}.png"))
    with open(os.path.join(output_dir, f"classification_result_{idx+1}.txt"), "w", encoding="utf-8") as f:
        f.write(f"True Label: {true_label}\n")
        f.write(f"Predicted: {pred_digit}\n")
        f.write(f"Correct: {pred_digit == true_label}\n")
print("✓ 样例生成完成，结果保存在 output_cls/ 目录中")



