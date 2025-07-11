from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

MODEL_NAME = "/root/autodl-fs/models/Qwen3-8B"
NUM_LABELS = 10

model = AutoModelForSequenceClassification.from_pretrained(
    "/root/autodl-fs/models/Qwen3-8B", 
    num_labels=NUM_LABELS, 
    device_map="auto", 
    torch_dtype="auto"
)

lora_cfg = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  

print("!!!!!")
print(isinstance(model, PeftModel))
model.save_pretrained("autodl-tmp/MLLM/models/peft_model")
print("~~~~~")

