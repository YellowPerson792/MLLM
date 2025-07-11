from transformers import TrainerCallback
from peft import PeftModel
import os

def byte_list_to_char_list(byte_list):
    return [chr(b) for b in byte_list]

def rename_column_byte_input(example):
    return {
        "text": byte_list_to_char_list(example["byte_array"]),  
        "label": example["label"]
    }
    
def rename_column(example):
    return {
        "text": "".join(byte_list_to_char_list(example["byte_array"])), 
        "label": example["label"]
    }
    
def rename_column_hex(example):
    return {
        "text": example["hex"],  
        "label": example["label"]
    }


class NStepsCallback(TrainerCallback):
    def __init__(self, save_every_steps=500, output_dir="./lora_checkpoints"):
        self.save_every_steps = save_every_steps
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_every_steps == 0:
            save_path = os.path.join(self.output_dir, f"step-{state.global_step}")
            os.makedirs(save_path, exist_ok=True)
            
            model = kwargs["model"]
            print("!!!!!")
            print(isinstance(model, PeftModel))
            print("~~~~~")
            # model.save_pretrained(save_path)
            
            # Optionally save tokenizer
            # tokenizer = kwargs.get("tokenizer")
            # if tokenizer:
            #     tokenizer.save_pretrained(save_path)

            control.should_save = True
        return control
