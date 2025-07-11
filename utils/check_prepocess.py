import os
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

# 输出目录
out_dir = '/root/autodl-tmp/MLLM/datasets/cifar10/images'
os.makedirs(out_dir, exist_ok=True)

# 预处理方式与训练/推理一致
preprocess = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(1.0, 1.0), ratio=(1.0, 1.0), antialias=True)
])

# Tokenizer 路径（与训练/推理保持一致）
MODEL_PATH = '/root/autodl-tmp/MLLM/models/jpeg-lm'
QUALITY = 25
MAX_SEQ_LEN = 4096

def convert_img_to_bytes(img: Image.Image, quality: int = QUALITY) -> str:
    img.save("cache_tables.jpg", format="JPEG", quality=quality, subsampling="4:2:0", streamtype=1, restart_marker_blocks=1)
    with os.fdopen(os.dup(os.open("cache_tables.jpg", os.O_RDONLY)), 'rb') as _:
        pass  # just to ensure file is closed after save
    import io
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG", quality=quality, subsampling="4:2:0", streamtype=2, restart_marker_blocks=1)
        data = buf.getvalue()
    return ''.join(chr(b + 10240) for b in data)

def tokenize_example(example, tokenizer):
    img = preprocess(example["img"])
    img = img.convert("RGB")
    jpeg_str = convert_img_to_bytes(img, QUALITY)
    toks = tokenizer(jpeg_str, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True)
    return toks["input_ids"], img

# 加载 MNIST
ds = load_dataset('uoft-cs/cifar10')
ds = ds.cast_column('img', ds['train'].features['img'])

# 选取前10个样本
examples = ds['train'].select(range(10))

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# 保存 JPEG 码流和 token 的文本文件
save_txt_path = '/root/autodl-tmp/MLLM/datasets/cifar10/samples.txt'
os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
with open(save_txt_path, 'w', encoding='utf-8') as ftxt:
    lengths = []
    for i, ex in enumerate(examples):
        input_ids, img_proc = tokenize_example(ex, tokenizer)
        save_path = os.path.join(out_dir, f'preprocessed_{i}_label{ex["label"]}.png')
        img_proc.save(save_path)
        print(f'Saved: {save_path}')
        # 统计有效 token 长度（去除 padding）
        valid_len = sum(1 for t in input_ids if t != tokenizer.pad_token_id)
        lengths.append(valid_len)
        # 保存 JPEG 码流和 token（码流用hex表示，token完整输出）
        img = preprocess(ex["img"]).convert("RGB")
        jpeg_bytes = img_proc.tobytes()  # 但这不是JPEG码流，需用convert_img_to_bytes
        jpeg_str = convert_img_to_bytes(img, QUALITY)
        jpeg_hex = ''.join(f'{ord(c)-10240:02x}' for c in jpeg_str)
        ftxt.write(f"Sample {i} label={ex['label']}\n")
        ftxt.write(f"JPEG hex: {jpeg_hex}\n")
        ftxt.write(f"Token ids: {input_ids}\n\n")

if lengths:
    avg_len = sum(lengths) / len(lengths)
    print(f"前10个样本的平均 token 长度: {avg_len:.2f}")
else:
    print("未统计到 token 长度")
