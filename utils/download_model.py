if __name__ == "__main__":
    # 示例：下载gpt2模型到默认缓存目录
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("正在下载gpt2模型...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("gpt2模型下载完成，已缓存到：", tokenizer.cache_dir if hasattr(tokenizer, 'cache_dir') else "默认transformers缓存目录")
