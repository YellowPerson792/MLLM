from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Qwen3ForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch 
import numpy as np
from transformers import DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForSequenceClassification, PeftModel, PeftConfig
from utils import *

MODEL_NAME = "Qwen/Qwen3-8B"
NUM_LABELS = 10

# model = AutoModelForSequenceClassification.from_pretrained(
#     "/root/autodl-tmp/MLLM/models_cache/Qwen/Qwen3-8B", 
#     num_labels=NUM_LABELS, 
#     device_map="auto", 
#     torch_dtype="auto"
# )

model = AutoPeftModelForSequenceClassification.from_pretrained(
    "/root/autodl-tmp/MLLM/test_folder/peft_model",
    num_labels=NUM_LABELS, 
    is_trainable=True,
    device_map="auto", 
    torch_dtype="auto"
)

# config = PeftConfig.from_pretrained("/root/autodl-tmp/MLLM/test_folder/loaded_peft_model")
# model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=NUM_LABELS)
# model = PeftModel.from_pretrained(model, "/root/autodl-tmp/MLLM/test_folder/loaded_peft_model", is_trainable=True)

# qwen3_lora_config = LoraConfig(
#     r=64,  # LoRA rank，常用 8/16/32/64，越大效果越好，但显存越高
#     lora_alpha=128,  # 一般设置为 2 * r
#     target_modules=["q_proj", "v_proj"],  # 只插入到注意力机制里，节省资源
#     lora_dropout=0.05,
#     bias="none",  # 不修改原始 bias
#     task_type=TaskType.SEQ_CLS,
# )

# model = get_peft_model(model, qwen3_lora_config)
model.print_trainable_parameters()  

print("!!!!!")
print(isinstance(model, PeftModel))
model.save_pretrained("/root/autodl-tmp/MLLM/test_folder/loaded_peft_model")
print("~~~~~")

