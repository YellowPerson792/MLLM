import os
import torch
from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig, GPT2Config, GPT2LMHeadModel
from jpeglm.models.jpeglm_encoder import create_jpeglm_encoder
from datasets import load_dataset
from utils.data_utils import convert_img_to_bytes, create_preprocess_transform
from peft import PeftModel
from PIL import Image

# 配置
encoder_path = "/root/autodl-fs/models/jpeg-lm"
decoder_path = "gpt2"
model_ckpt = "/root/autodl-tmp/MLLM/checkpoints/jpeglm-gpt2-mnist-classification/checkpoint-2560"  # 你的模型保存路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载分词器
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path)
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_path)
decoder_tokenizer.pad_token = decoder_tokenizer.unk_token

# 加载模型
gpt2_config = GPT2Config.from_pretrained(decoder_path)
gpt2_config.add_cross_attention = True
gpt2 = GPT2LMHeadModel.from_pretrained(decoder_path, config=gpt2_config)
encoder = create_jpeglm_encoder(encoder_path)
base_model = EncoderDecoderModel(encoder=encoder, decoder=gpt2)
base_model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
base_model.config.pad_token_id = decoder_tokenizer.pad_token_id
base_model.config.eos_token_id = decoder_tokenizer.eos_token_id
base_model.config.vocab_size = base_model.config.decoder.vocab_size
base_model.main_input_name = "input_ids"
generation_config = GenerationConfig(
    max_length=18,
    num_beams=1,
    no_repeat_ngram_size=3,
    decoder_start_token_id=base_model.config.decoder_start_token_id,
    bos_token_id=decoder_tokenizer.bos_token_id,
    pad_token_id=decoder_tokenizer.pad_token_id,
    eos_token_id=decoder_tokenizer.eos_token_id,
)
base_model.generation_config = generation_config

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, model_ckpt)
model = model.to(device)
model.eval()

# 标签映射
digit_to_text = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}
text_to_digit = {v: k for k, v in digit_to_text.items()}

def extract_digit_from_text(text):
    import re
    digit_words = text_to_digit
    text = text.lower()
    for word, digit in digit_words.items():
        if word in text:
            return digit
    digits = re.findall(r'\\d', text)
    if digits:
        return int(digits[0])
    return None

# 加载MNIST测试集
mnist_dataset = load_dataset("ylecun/mnist")
test_data = mnist_dataset["test"].select(range(20))  # 只推理前20条，可自行调整

transform = create_preprocess_transform(96)

print("==== 推理结果 ====")
for idx, item in enumerate(test_data):
    pil_image = item['image'].convert("RGB")
    label = item['label']
    # 预处理
    img = transform(pil_image)
    jpeg_str = convert_img_to_bytes(img)
    input_ids = [encoder_tokenizer.bos_token_id] + encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
    input_ids = input_ids[:1024]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(inputs=input_ids, generation_config=generation_config)
    generated_text = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    pred_digit = extract_digit_from_text(generated_text)
    print(f"[{idx}] True: {label} | Gen: {generated_text} | Pred: {pred_digit} | {'✓' if pred_digit==label else '✗'}")

print("推理结束。")