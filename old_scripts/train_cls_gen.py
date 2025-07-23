import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# === Configuration ===
REPORT_TO_WANDB = True
MODEL_DIR = "/root/autodl-tmp/MLLM/models/Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "/root/autodl-tmp/MLLM/datasets/mnist/mnist_jpeg.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/MLLM/checkpoints"
PROMPT = "请判断以下文本的类别（0-9）："
MAX_LENGTH = 1024
TEST_SPLIT_RATIO = 0.1
BATCH_SIZE_TRAIN = 1
BATCH_SIZE_EVAL = 2
GRADIENT_ACCUM_STEPS = 4
NUM_EPOCHS = 2
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
USE_WANDB = REPORT_TO_WANDB


def preprocess_fn(example, tokenizer):
    """
    Tokenize and prepare inputs, attention masks, and labels.
    """
    # Build instruction + user prompt
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['hex']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    # Tokenize the label text
    response = tokenizer(str(example['label']), add_special_tokens=False)

    # Combine tokens
    input_ids = instruction['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = instruction['attention_mask'] + response['attention_mask'] + [1]
    # Set loss mask: ignore system+prompt and pad
    labels = [-100] * len(instruction['input_ids']) + response['input_ids'] + [-100]

    # Truncate if needed
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def load_peft_model(model_dir: str, checkpoint_dir: str, tokenizer):
    """
    Load base model and apply LoRA weights.
    """
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    base_model.enable_input_require_grads()

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=TARGET_MODULES,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    # Apply PEFT
    model = get_peft_model(base_model, lora_config)
    return model


def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)
    model = load_peft_model(MODEL_DIR, OUTPUT_DIR, tokenizer)

    # Load and split dataset
    dataset = load_dataset('json', data_files=DATA_FILE, split='train')
    splits = dataset.train_test_split(test_size=TEST_SPLIT_RATIO, seed=42)

    train_ds = splits['train'].map(
        lambda ex: preprocess_fn(ex, tokenizer),
        remove_columns=splits['train'].column_names
    )
    eval_ds = splits['test'].map(
        lambda ex: preprocess_fn(ex, tokenizer),
        remove_columns=splits['test'].column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        eval_strategy='epoch',
        logging_steps=20,
        save_steps=100,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        save_on_each_node=True,
        gradient_checkpointing=True,
        save_total_limit=3,
        report_to='wandb' if USE_WANDB else 'none',
        run_name=f"{os.path.basename(MODEL_DIR)}-lora-{os.path.basename(DATA_FILE)}",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
