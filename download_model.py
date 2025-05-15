from huggingface_hub import snapshot_download

# 替换为你要下载的模型名，如 "meta-llama/Llama-2-7b-hf"
repo_id = "Qwen/Qwen3-8B"

# 替换为你希望保存的本地路径
local_dir = "/root/autodl-tmp/MLLM/models_cache/Qwen/Qwen3-8B"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
)
