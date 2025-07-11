from datasets import load_dataset
import io
import json
import os
import binascii
import random
import struct

from PIL import Image
from os.path import basename

DATASET = "ylecun/mnist"
dataset = load_dataset(DATASET, split="train")
dataset = dataset.shuffle(seed=42)

def extract_huffman_tables(jpeg_bytes):
    """
    从 JPEG bytes 中解析所有 DHT 段，返回一个 dict:
      { (tc, th) → { symbol: (code_bits_str, length) } }
    """
    tables = {}
    i = 0
    while i < len(jpeg_bytes):
        if jpeg_bytes[i] == 0xFF and jpeg_bytes[i+1] == 0xC4:
            length = struct.unpack(">H", jpeg_bytes[i+2:i+4])[0]
            segment = jpeg_bytes[i+4 : i+2+length]
            ptr = 0
            while ptr < len(segment):
                tc_th = segment[ptr]; ptr += 1
                tc = tc_th >> 4
                th = tc_th & 0x0F
                counts = list(segment[ptr:ptr+16]); ptr += 16
                symbols = list(segment[ptr:ptr+sum(counts)]); ptr += sum(counts)
                code = 0
                huffmap = {}
                for bits_len in range(1,17):
                    for _ in range(counts[bits_len-1]):
                        sym = symbols.pop(0)
                        code_bits = format(code, f'0{bits_len}b')
                        huffmap[sym] = (code_bits, bits_len)
                        code += 1
                    code <<= 1
                tables[(tc,th)] = huffmap
            i += 2 + length
        else:
            i += 2 if jpeg_bytes[i] == 0xFF else 1
    return tables

def bits_to_hex_chunks(bitstr, sep_bits, pad='0'):
    """
    将一个大 bitstr 按 sep_bits 分割后，
    把每段补齐到字节边界，再转换成 hex 字符串列表。
    """
    parts = bitstr.split(sep_bits)
    hex_chunks = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part_bits = part + sep_bits
        else:
            part_bits = part
        # 补齐到 8 的倍数
        if len(part_bits) % 8 != 0:
            part_bits += pad * (8 - len(part_bits) % 8)
        # 每 8 位为一字节
        byts = bytes(int(part_bits[j:j+8], 2) for j in range(0, len(part_bits), 8))
        hex_chunks.append(binascii.hexlify(byts).decode('utf-8'))
    return hex_chunks

def split_entropy_by_eob(hex_str):
    data = binascii.unhexlify(hex_str)
    # 定位 SOS 和 EOI
    sos_off = data.find(b'\xFF\xDA')
    eoi_off = data.rfind(b'\xFF\xD9')
    if sos_off < 0 or eoi_off < 0:
        return hex_str  # 非标准 JPEG
    sos_len = struct.unpack(">H", data[sos_off+2:sos_off+4])[0]
    header = data[:sos_off+2+sos_len]
    entropy = data[sos_off+2+sos_len : eoi_off]
    trailer = data[eoi_off:]

    # 提取 Huffman 表，取 AC (tc=1) 表 0
    tables = extract_huffman_tables(data)
    ac_table = tables.get((1,0))
    if ac_table is None or 0 not in ac_table:
        return hex_str

    eob_bits, _ = ac_table[0]  # EOB 对应的比特码字

    # 展开 entropy 为 bit 流，去掉 0xFF00 填充
    bits = []
    i = 0
    while i < len(entropy):
        b = entropy[i]
        if b == 0xFF and i+1 < len(entropy) and entropy[i+1] == 0x00:
            bits.append('11111111')
            i += 2
        else:
            bits.append(f'{b:08b}')
            i += 1
    bitstr = ''.join(bits)

    # 按 EOB 拆分并转成 hex 块
    hex_chunks = bits_to_hex_chunks(bitstr, eob_bits)

    # 重新拼接：header_hex + 空格分隔的 entropy_chunks + trailer_hex
    return (
        binascii.hexlify(header).decode('utf-8')
        + ' '
        + ' '.join(hex_chunks)
        + ' '
        + binascii.hexlify(trailer).decode('utf-8')
    )

def convert(example):
    buffer = io.BytesIO()
    example["image"].save(buffer,
                         format="JPEG",
                         quality=90,
                         optimize=False,
                         progressive=False)
    byte_data = buffer.getvalue()
    hex_all = binascii.hexlify(byte_data).decode("utf-8")
    example["hex"] = hex_all
    example["hex_eob"] = split_entropy_by_eob(hex_all)
    return example

# 转换数据集
dataset = dataset.map(convert, remove_columns=["image"])

# 找最长公共前缀
def longest_common_prefix(strs):
    if not strs: return ''
    shortest = min(strs, key=len)
    for i, ch in enumerate(shortest):
        if any(other[i] != ch for other in strs):
            return shortest[:i]
    return shortest

all_vals = [ex["hex_eob"] for ex in dataset]
prefix = longest_common_prefix(all_vals)
print(f"公共前缀长度: {len(prefix)}")

REMOVE_PREFIX = True
output_dir = f"/root/autodl-tmp/MLLM/datasets/{basename(DATASET)}"
os.makedirs(output_dir, exist_ok=True)

def write_jsonl(examples, path):
    with open(path, "w") as f:
        f.write('[\n')
        for idx, ex in enumerate(examples):
            val = ex["hex_eob"][len(prefix):] if REMOVE_PREFIX else ex["hex_eob"]
            item = {
                "instruction": "请判断以下比特流图片的类别（0-9）",
                "input": val,
                "output": str(ex["label"]),
                "system": ""
            }
            line = json.dumps(item, ensure_ascii=False)
            f.write(line)
            f.write(',\n' if idx < len(examples)-1 else '\n')
        f.write(']\n')

# 写入全量数据集
full_path = os.path.join(output_dir, f"{basename(DATASET)}_jpeg_eob_split.jsonl")
write_jsonl(dataset, full_path)

# 写入 small 数据集 (随机 1000 条)
small_indices = random.sample(range(len(dataset)), 1000)
small_ds = dataset.select(small_indices)
small_path = os.path.join(output_dir, f"{basename(DATASET)}_jpeg_eob_small.jsonl")
write_jsonl(small_ds, small_path)

print("Done!")
