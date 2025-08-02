# 示例运行命令：
# python ImageCaption/train_vit-gpt2_cls.py --train_batch_size 8 --eval_batch_size 8 --eval_strategy steps --eval_steps 128 --logging_steps 128 --save_steps 512 --learning_rate 5e-5 --num_train_epochs 3 --save_total_limit 1 --lr_scheduler_type linear --gradient_accumulation_steps 1 --report_to None --num_train_samples 6000 --num_eval_samples 16

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_style_trainer import MySeq2SeqTrainingArguments
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import EncoderDecoderModel, GPT2LMHeadModel, ViTFeatureExtractor, ViTModel, AutoTokenizer, GenerationConfig, GPT2Config
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import datasets
from peft import get_peft_model, LoraConfig, TaskType

# 自定义ViT包装类，兼容EncoderDecoderModel的输入参数
class ViTEncoderWrapper(ViTModel):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        return super().forward(pixel_values=pixel_values, **{k: v for k, v in kwargs.items() if k in ['head_mask', 'output_attentions', 'output_hidden_states', 'return_dict']})


import numpy as np
import torch
import multiprocessing as mp
import argparse
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import EncoderDecoderModel, GPT2LMHeadModel, ViTFeatureExtractor, ViTModel, AutoTokenizer, GenerationConfig, GPT2Config
from sklearn.model_selection import train_test_split
from hf_style_trainer import MySeq2SeqTrainingArguments
import datasets

# 配置
class config:
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    SEED = 42
    IMG_SIZE = 32
    MAX_LEN = 1
    NUM_WORKERS = mp.cpu_count()
os.environ["WANDB_DISABLED"] = "true"
script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizer/FeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
decoder_tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
decoder_tokenizer.pad_token = decoder_tokenizer.unk_token

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="./vit-gpt2-mnist-cls")
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
parser.add_argument('--num_train_samples', type=int, default=6000)
parser.add_argument('--num_eval_samples', type=int, default=1000)
args = parser.parse_args()

# 加载MNIST
from datasets import load_dataset
print("正在加载MNIST数据集...")
mnist_dataset = load_dataset("ylecun/mnist")
train_data = mnist_dataset["train"]
test_data = mnist_dataset["test"]

# 按参数裁剪数据集
train_data = train_data.select(range(min(args.num_train_samples, len(train_data))))
test_data = test_data.select(range(min(args.num_eval_samples, len(test_data))))
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

digit_to_text = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

class MNISTViTDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor, decoder_tokenizer, digit_to_text, image_size=32, max_length=20):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.decoder_tokenizer = decoder_tokenizer
        self.digit_to_text = digit_to_text
        self.image_size = image_size
        self.max_length = max_length


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = item['label']
        caption = self.digit_to_text[label]
        img = item['image'].convert("RGB").resize((self.image_size, self.image_size))
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values.squeeze()
        labels = self.decoder_tokenizer(
            caption,
            max_length=self.max_length,
            truncation=True
        ).input_ids
        labels = [token if token != self.decoder_tokenizer.pad_token_id else -100 for token in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


# 构建数据集
train_dataset = MNISTViTDataset(train_data, feature_extractor, decoder_tokenizer, digit_to_text, image_size=config.IMG_SIZE, max_length=config.MAX_LEN)
val_dataset = MNISTViTDataset(test_data, feature_extractor, decoder_tokenizer, digit_to_text, image_size=config.IMG_SIZE, max_length=config.MAX_LEN)

# 动态padding的collate_fn
def dynamic_pad_collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    pixel_values = torch.stack(pixel_values)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": labels}

# 构建EncoderDecoderModel
gpt2_config = GPT2Config.from_pretrained(config.DECODER)
gpt2_config.add_cross_attention = True
gpt2 = GPT2LMHeadModel.from_pretrained(config.DECODER, config=gpt2_config)
vit = ViTEncoderWrapper.from_pretrained(config.ENCODER)
model = EncoderDecoderModel(encoder=vit, decoder=gpt2)
model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
model.config.pad_token_id = decoder_tokenizer.pad_token_id
model.config.eos_token_id = decoder_tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.main_input_name = "pixel_values"

# GenerationConfig
generation_config = GenerationConfig(
    max_new_tokens=config.MAX_LEN,
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

def extract_digit_from_text(text):
    import re
    digit_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    text = text.lower()
    for word, digit in digit_words.items():
        if word in text:
            return digit
    digits = re.findall(r'\d', text)
    if digits:
        return int(digits[0])
    return None

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    import torch
    from torch.nn.utils.rnn import pad_sequence
    labels_ids = [torch.tensor(x) for x in labels_ids]
    pred_ids = [torch.tensor(x) for x in pred_ids]
    labels_ids = pad_sequence(labels_ids, batch_first=True, padding_value=decoder_tokenizer.pad_token_id)
    pred_ids = pad_sequence(pred_ids, batch_first=True, padding_value=decoder_tokenizer.pad_token_id)
    pred_str = decoder_tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
    labels_ids[labels_ids == -100] = decoder_tokenizer.pad_token_id
    label_str = decoder_tokenizer.batch_decode(labels_ids.tolist(), skip_special_tokens=True)
    correct = 0
    total = 0
    for pred_text, label_text in zip(pred_str, label_str):
        pred_digit = extract_digit_from_text(pred_text)
        label_digit = extract_digit_from_text(label_text)
        if pred_digit is not None and label_digit is not None:
            if pred_digit == label_digit:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0.0
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
        "accuracy": round(accuracy, 4),
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

# ====== 使用LoRA进行参数高效训练 ======
# 自动收集所有decoder.transformer.h的子模块名
# h_modules = [f"decoder.transformer.h.{i}" for i in range(model.decoder.config.n_layer)]
h_modules = []
modules_to_save = h_modules + [
    # "decoder.transformer.ln_f",
    "mlp.c_fc",
    "mlp.c_proj",
    "crossattention.q_attn",
    "crossattention.c_attn",
    "crossattention.c_proj",
    "decoder.lm_head",
    "enc_to_dec_proj"
]
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # 可调
    lora_alpha=32,  # 可调
    target_modules = [
    # — ViT Encoder Self-Attention
    "attention.attention.query",       # q_proj
    "attention.attention.value",       # v_proj
    "attention.output.dense",          # o_proj
    # — ViT Encoder MLP
    "intermediate.dense",              # up_proj / gate_proj
    "output.dense",                    # down_proj
    # # — GPT-2 Decoder Self-Attention (c_attn 包含 Q/K/V)
    # "attn.c_attn",                     # slice 出 q_proj/v_proj
    # # — GPT-2 Decoder Cross-Attention
    # "crossattention.q_attn",           # decoder→encoder 的 query
    # "crossattention.c_attn",           # encoder 输出做 key/value
    # # — GPT-2 Decoder MLP
    # "mlp.c_fc",                        # up_proj / gate_proj
    # "mlp.c_proj",                      # down_proj
    # # （可选）如果想针对输出投影再加一次 LoRA：
    # "attn.c_proj",                     # self-attention 最后一段投影
],
    modules_to_save=modules_to_save,
    lora_dropout=0.1
)

# 开启梯度检查点
# model.gradient_checkpointing_enable()
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
    img = pil_image.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    pixel_values = feature_extractor(img, return_tensors="pt").pixel_values.to(model.device)
    generated_ids = model.generate(pixel_values=pixel_values, generation_config=generation_config)
    generated_text = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

print("\n==== 生成分类样例 ====")
for idx in range(3):
    sample = test_data[idx]
    pil_image = sample['image']
    true_label = sample['label']
    true_caption = digit_to_text[true_label]
    generated_caption = generate_classification_from_image(pil_image)
    print(f"Sample {idx + 1}:")
    print(f"True Label: {true_label}")
    print(f"True Caption: {true_caption}")
    print(f"Generated Caption: {generated_caption}")
    pred_digit = extract_digit_from_text(generated_caption)
    print(f"Predicted Digit: {pred_digit}")
    print(f"Correct: {pred_digit == true_label}")
    print("-" * 50)
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_img_path = os.path.join(output_dir, f"mnist_sample_{idx+1}.png")
    pil_image.save(save_img_path)
    result_path = os.path.join(output_dir, f"classification_result_{idx+1}.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"True Label: {true_label}\n")
        f.write(f"True Caption: {true_caption}\n")
        f.write(f"Generated Caption: {generated_caption}\n")
        f.write(f"Predicted Digit: {pred_digit}\n")
        f.write(f"Correct: {pred_digit == true_label}\n")
print("✓ 样例生成完成，结果保存在 output/ 目录中")
