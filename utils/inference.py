import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==== Configuration (define parameters here) ====
MODEL_DIR = "/root/autodl-fs/models/Qwen2.5-7B-Instruct"  # Base model directory
LORA_DIR = "/root/autodl-fs/models/checkpoint-9500"  # Set None to skip LoRA
MODE = "base"  # Options: "base" or "lora"
HEX = "90001010003010000000000000000000000000704050609ffc400251000010304020202030100000000000000010203040005061107121321084122314271ffda0008010100003f00f2aaa8b87e19c7990351d8baf20ceb05cde532801dc78bd1105493dfb3a87fb8095751b0d1dfb3eb437cd720615378e734bc633717e2499b6c90a8eebd05f4bcca88fb4a87fbec1d107608041039fa55bbe326232afd3afd71c61a8578e48b736d1c76c535e6daeeb59525c98df974879d63f0f1b2141656e21c0141a524c7efec5ca35f2e0d5e5125bbba5f589899a141e0f763dfbf6f7dbb6f7bf7bac1a551708e06ceb33b2c6c96db6a542c67cea6d7924e7d31a04552012a2e3ea20235d4ebed44692147d56e7e54e7f8ef24f2ebf76c6cae5476add060cbbb2dbf11bacb623a1a7e678ff80e2d0481fb2344e8922a434a5295ffd9"
PROMPT_INPUT = f"请判断以下比特流图片的类别（0-9）\n"\
                f"{HEX}<|im_end|>\n"  # Any user input
# PROMPT_INPUT = "介绍一下你自己"
FILE_INPUT = None      # Or set to "/path/to/your/file.txt" for batch inputs
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."  # If you want a system prompt, set like "你是一个友好的助手。"
MAX_NEW_TOKENS = 200   # Number of tokens to generate for a full reply
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_base_model(model_dir: str, device: str):
    """
    Load and return the base tokenizer and model (e.g., Qwen3-8B).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    base_model.to(device)
    base_model.eval()
    return tokenizer, base_model


def apply_lora_adapter(base_model, lora_dir: str):
    """
    Apply LoRA adapter weights to an existing base model.
    """
    if not lora_dir:
        raise ValueError("LORA_DIR must be set when MODE is 'lora'.")
    print(f"Applying LoRA adapter from {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.to(base_model.device)
    model.eval()
    return model


def build_prompt(user_input: str, system_prompt: str = None) -> str:
    """
    Construct prompt for generation. If system_prompt is provided, wrap in chat template.
    Else use user_input directly.
    """
    if system_prompt:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return user_input


def generate_full_response(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    """
    Generate a full response from the model given a prompt.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9
    )
    # Slice off prompt tokens, decode the rest
    gen_ids = output[0][ inputs.input_ids.shape[-1] : ].cpu().tolist()
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return decoded


def main():
    # Load base model
    tokenizer, base_model = init_base_model(MODEL_DIR, DEVICE)

    # Choose model mode
    if MODE == 'lora':
        if LORA_DIR is None:
            raise ValueError("LORA_DIR must be set when MODE is 'lora'.")
        print("Running in LoRA adapter mode")
        model = apply_lora_adapter(base_model, LORA_DIR)
    elif MODE == 'base':
        print("Running in base model mode (no LoRA adapter)")
        model = base_model
    else:
        raise ValueError("MODE must be 'base' or 'lora'")

    # Prepare inputs
    samples = []
    if PROMPT_INPUT:
        samples.append(PROMPT_INPUT)
    if FILE_INPUT:
        with open(FILE_INPUT, 'r') as f:
            samples.extend([line.strip() for line in f if line.strip()])
    if not samples:
        raise ValueError("No input provided: set PROMPT_INPUT or FILE_INPUT.")

    # Run inference for each sample
    for idx, user_input in enumerate(samples, 1):
        prompt = build_prompt(user_input, SYSTEM_PROMPT)
        print(f"[Sample {idx}/{len(samples)}] 输入到模型的文本如下：\n{prompt}\n")
        response = generate_full_response(
            tokenizer, model, prompt, MAX_NEW_TOKENS
        )
        print(f"[Sample {idx}/{len(samples)}] User Input: {user_input}")
        print(f"--- Model Response ---\n{response}\n")

if __name__ == '__main__':
    main()
