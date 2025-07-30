from transformers import AutoModelForCausalLM
import safetensors.torch

# 1. 加载完整的模型（含 lm_head）
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-fs/models/jpeg-lm",
)

# 2. 把 state_dict（包含所有参数和 buffer）导出为 safetensors
safetensors.torch.save_file(model.state_dict(), "model.safetensors")
