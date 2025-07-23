import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# === 配置 ===
CHECKPOINT_DIR = "/root/autodl-fs/models/checkpoint-9500"
DATA_FILE      = "/root/autodl-tmp/MLLM/datasets/mnist/mnist_jpeg-test.jsonl"
MAX_LENGTH     = 1200
PROMPT         = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer + 模型
tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-fs/models/Qwen2.5-7B-Instruct",
    use_fast=False,
    trust_remote_code=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-fs/models/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.to(DEVICE).eval()

def generate_label(hex_text: str) -> str:
    """给定一段 hex 文本，返回模型预测的单个类别字符。"""
    instr = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"请判断以下比特流图片的类别（0-9）\n"
        f"{hex_text}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        return_tensors="pt",
        add_special_tokens=False,
    ).to(DEVICE)

    out = model.generate(
        instr.input_ids,
        max_new_tokens=4,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][ instr.input_ids.shape[-1] : ].cpu().tolist()
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return pred[:1] if pred else ""

def main():
    # 载入并拆分数据
    raw = load_dataset("json", data_files=DATA_FILE, split="train")
    # splits = raw.train_test_split(test_size=0.1, seed=42)
    # raw_test = splits["test"]
    raw_test = raw
    total = len(raw_test)

    correct = 0
    processed = 0
    # 新增：统计每个类别的正确数和总数
    class_correct = {str(i): 0 for i in range(10)}
    class_total = {str(i): 0 for i in range(10)}
    # 新增：统计模型输出各数字的次数
    pred_count = {str(i): 0 for i in range(10)}

    print(f"测试集共 {total} 条样本，按 Ctrl+C 随时中断并查看当前准确率。\n")

    try:
        for ex in tqdm(raw_test, total=total, desc="Evaluating"):
            true_label = str(ex["label"])
            pred_label = generate_label(ex["hex"])
            processed += 1
            class_total[true_label] += 1
            if pred_label == true_label:
                correct += 1
                class_correct[true_label] += 1
            if pred_label in pred_count:
                pred_count[pred_label] += 1

            input_excerpt = ex["hex"][:100] + ("..." if len(ex["hex"])>100 else "")
            current_acc = correct / processed if processed > 0 else 0.0
            tqdm.write(f"[{processed}/{total}] Input: {input_excerpt}")
            tqdm.write(f"    True: {true_label}   Pred: {pred_label}   Acc: {correct}/{processed} = {current_acc:.4f}")
            # 每条都输出各类别准确率
            class_accs = []
            for i in range(10):
                label = str(i)
                total_i = class_total[label]
                correct_i = class_correct[label]
                acc_i = correct_i / total_i if total_i > 0 else 0.0
                class_accs.append(f"{label}:{correct_i}/{total_i}={acc_i:.3f}")
            tqdm.write("    Class Accs: " + "  ".join(class_accs))
            # 新增：输出模型输出各数字的百分比
            pred_percent = []
            for i in range(10):
                label = str(i)
                percent = pred_count[label] / processed if processed > 0 else 0.0
                pred_percent.append(f"{label}:{pred_count[label]}/{processed}={percent:.3f}")
            tqdm.write("    Pred Dist: " + "  ".join(pred_percent) + "\n")

    except KeyboardInterrupt:
        print("\n检测到中断，准备输出当前结果…")

    if processed == 0:
        print("未处理任何样本。")
    else:
        final_acc = correct / processed
        print(f"\n已处理 {processed} 条样本，准确率：{correct}/{processed} = {final_acc:.4f}")
        print("各类别准确率：")
        for i in range(10):
            label = str(i)
            total_i = class_total[label]
            correct_i = class_correct[label]
            acc_i = correct_i / total_i if total_i > 0 else 0.0
            print(f"  类别 {label}: {correct_i}/{total_i} = {acc_i:.4f}")
        print("模型输出分布：")
        for i in range(10):
            label = str(i)
            percent = pred_count[label] / processed if processed > 0 else 0.0
            print(f"  输出 {label}: {pred_count[label]}/{processed} = {percent:.4f}")

if __name__ == "__main__":
    main()
