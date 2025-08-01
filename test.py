from transformers import AutoTokenizer

# 加载gpt2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("原始特殊token：")
print("pad_token:", tokenizer.pad_token)
print("unk_token:", tokenizer.unk_token)
print("eos_token:", tokenizer.eos_token)
print("bos_token:", tokenizer.bos_token)

# 设置pad_token为unk_token
tokenizer.pad_token = tokenizer.unk_token

print("当前特殊token：")
print("pad_token:", tokenizer.pad_token)
print("unk_token:", tokenizer.unk_token)
print("eos_token:", tokenizer.eos_token)
print("bos_token:", tokenizer.bos_token)

# 测试文本
text = "hello world"
print("\n编码效果：")
print("add_special_tokens=False:", tokenizer.encode(text, add_special_tokens=False))
print("add_special_tokens=True :", tokenizer.encode(text, add_special_tokens=True))

print("\n解码效果：")
print("add_special_tokens=False:", tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)))
print("add_special_tokens=True :", tokenizer.decode(tokenizer.encode(text, add_special_tokens=True)))
