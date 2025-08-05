# 示例运行命令：
# python prefix_gpt2_prefixgen.py --train_batch_size 2 --eval_batch_size 2 --num_train_epochs 5 --learning_rate 2e-4 --output_dir ./checkpoints/prefix-gpt2-prefixgen

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
import argparse
from hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="./checkpoints/prefix-gpt2-prefixgen")
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--num_train_epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--eval_strategy', type=str, default="epoch")
parser.add_argument('--eval_steps', type=int, default=200)
parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--save_steps', type=int, default=200)
parser.add_argument('--save_total_limit', type=int, default=2)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--lr_scheduler_type', type=str, default="linear")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--report_to', type=str, default=None)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--bf16', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构造自定义前缀-目标对
prefix_target_pairs = [
    ("A:", "Apple"),
    ("B:", "Banana"),
    ("C:", "Cat"),
    ("D:", "Dog"),
    ("E:", "Egg"),
    ("F:", "Fish"),
    ("G:", "Goat"),
    ("H:", "Hat"),
    ("I:", "Ice"),
    ("J:", "Jam"),
]

# 加载tokenizer和模型
decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")
decoder_tokenizer.pad_token = decoder_tokenizer.unk_token

# 将目标单词转换为GPT2 tokenizer中的token IDs
gpt2_token_ids = {}
for _, word in prefix_target_pairs:
    token_ids = decoder_tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) == 1:
        gpt2_token_ids[word] = token_ids[0]
    else:
        print(f"警告: '{word}' 被tokenize为多个token: {token_ids}")
        gpt2_token_ids[word] = token_ids[0]  # 取第一个token

gpt2_config = AutoConfig.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 生成prefix embedding层 (简化的encoder)
prefix_embedder = torch.nn.Embedding(len(prefix_target_pairs), gpt2_config.n_embd).to(device)

# PrefixLM模型封装
class PrefixLMForPrefixGen(PreTrainedModel, GenerationMixin):
    supports_gradient_checkpointing = True
    main_input_name = "input_ids"
    base_model_prefix = "prefix_lm"

    def __init__(self, prefix_embedder, decoder, decoder_tokenizer, bos_token_id):
        # 使用decoder的config初始化PreTrainedModel
        super().__init__(decoder.config)
        self.prefix_embedder = prefix_embedder
        self.decoder = decoder
        self.decoder_tokenizer = decoder_tokenizer
        self.bos_token_id = bos_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        prefix_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取prefix embeddings
        if input_ids is not None:
            prefix_embeds = self.prefix_embedder(input_ids)  # [batch, hidden]
            prefix_embeds = prefix_embeds.unsqueeze(1)  # [batch, 1, hidden]
        else:
            # 如果没有input_ids，使用默认的零向量
            batch_size = labels.size(0) if labels is not None else 1
            prefix_embeds = torch.zeros(batch_size, 1, self.config.n_embd, device=labels.device if labels is not None else self.decoder.device)

        # 直接将prefix_embeds与labels的embeds拼接
        if labels is not None:
            # labels为token id序列，获取其embedding
            label_embeds = self.decoder.transformer.wte(labels)
            final_inputs_embeds = torch.cat([prefix_embeds, label_embeds], dim=1)
        elif decoder_inputs_embeds is not None:
            final_inputs_embeds = torch.cat([prefix_embeds, decoder_inputs_embeds], dim=1)
        else:
            final_inputs_embeds = prefix_embeds

        # Decode，确保不会同时传入input_ids和inputs_embeds
        decoder_args = {
            'attention_mask': decoder_attention_mask,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'use_cache': use_cache,
            'past_key_values': past_key_values,
            'return_dict': return_dict,
            **kwargs,
        }
        
        # 优先使用inputs_embeds（包含prefix）
        if final_inputs_embeds is not None:
            decoder_args['inputs_embeds'] = final_inputs_embeds
        elif decoder_input_ids is not None:
            decoder_args['input_ids'] = decoder_input_ids
        decoder_outputs = self.decoder(**decoder_args)

        # Compute loss independent from decoder
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            # 取prefix后第一个token的预测logits
            prediction_logits = logits[:, 0, :]  # [batch, vocab_size]
            # 展平
            shift_logits = prediction_logits.contiguous().view(-1, prediction_logits.size(-1))
            shift_labels = labels.contiguous().view(-1)
            pred = torch.argmax(shift_logits, dim=-1)
            # print(f"[DEBUG] pred: {pred}, shift_labels: {shift_labels}")
            loss = CrossEntropyLoss()(shift_logits, shift_labels)

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs
            else:
                return decoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=getattr(decoder_outputs, 'cross_attentions', None),
        )
        
    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

# 数据集
class PrefixGenDataset(Dataset):
    def __init__(self, pairs, decoder_tokenizer, gpt2_token_ids, max_length=32):
        self.pairs = pairs
        self.decoder_tokenizer = decoder_tokenizer
        self.gpt2_token_ids = gpt2_token_ids
        self.max_length = max_length
        # 创建prefix到id的映射
        self.prefix_to_id = {pair[0]: i for i, pair in enumerate(pairs)}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prefix, target = self.pairs[idx]
        # 使用GPT2 tokenizer中的真实token ID
        label_token_id = self.gpt2_token_ids[target]
        prefix_id = self.prefix_to_id[prefix]
        return {
            "prefix_id": torch.tensor(prefix_id),
            "label_id": torch.tensor(label_token_id)
        }

# 加载数据集
train_dataset = PrefixGenDataset(prefix_target_pairs, decoder_tokenizer, gpt2_token_ids)
val_dataset = PrefixGenDataset(prefix_target_pairs, decoder_tokenizer, gpt2_token_ids)

def collate_fn(batch):
    prefix_ids = [item["prefix_id"] for item in batch]
    label_ids = [item["label_id"] for item in batch]
    prefix_ids = torch.stack(prefix_ids)
    label_ids = torch.stack(label_ids)
    label_ids = label_ids.unsqueeze(1)  # [batch, 1]
    return {"input_ids": prefix_ids, "labels": label_ids}

# MySeq2SeqTrainer集成
bos_token_id = decoder_tokenizer.bos_token_id or decoder_tokenizer.eos_token_id or 50256

# 在模型外部确保特殊token id全部设置到config
model = PrefixLMForPrefixGen(prefix_embedder, gpt2_model, decoder_tokenizer, bos_token_id)
model.config.decoder_start_token_id = bos_token_id
model.config.pad_token_id = decoder_tokenizer.pad_token_id or decoder_tokenizer.unk_token_id

my_args = MySeq2SeqTrainingArguments(
    output_dir=args.output_dir,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    save_total_limit=args.save_total_limit,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    eval_strategy=args.eval_strategy,
    warmup_steps=args.warmup_steps,
    lr_scheduler_type=args.lr_scheduler_type,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    report_to=args.report_to if args.report_to not in [None, "None"] else None,
    fp16=args.fp16,
    bf16=args.bf16,
)

###############################################################
# 分类模式专用Trainer，重写evaluate方法
###############################################################
class PrefixGenTrainer(MySeq2SeqTrainer):
    def evaluate(self, eval_dataset=None, desc="Eval", ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        self.model.eval()
        device = self.args.device if hasattr(self.args, 'device') else self.model.device
        all_preds = []
        all_labels = []
        total_loss = 0.0
        debug_print_samples = []
        with torch.no_grad():
            for batch in tqdm(eval_dataset, desc=f"{desc} (custom)"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                preds = logits.argmax(dim=-1)[:, 0]  # 取第一个位置的预测
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.squeeze(-1).detach().cpu())
                total_loss += loss.item() * labels.size(0)
                # 收集前10条数据的预测和真实token及解码，全部预测完后统一打印
                if len(debug_print_samples) < 10:
                    for i in range(min(labels.size(0), 10 - len(debug_print_samples))):
                        pred_token = preds[i].item()
                        label_token = labels[i].squeeze(-1).item()
                        pred_word = self.tokenizer.decode([pred_token]) if self.tokenizer is not None else str(pred_token)
                        label_word = self.tokenizer.decode([label_token]) if self.tokenizer is not None else str(label_token)
                        debug_print_samples.append((pred_word, pred_token, label_word, label_token))
        # 拼接所有预测和标签，直接计算准确率
        all_preds = torch.cat(all_preds, dim=0).numpy()  # [N] - token ids
        all_labels = torch.cat(all_labels, dim=0).numpy()  # [N]
        # 直接计算准确率，不再调用compute_metrics
        correct = (all_preds == all_labels).sum()
        total = all_labels.shape[0]
        # 修正准确率计算，确保为比例（0~1），不是数量
        accuracy = float(correct) / float(total) if total > 0 else 0.0
        metrics = {"accuracy": round(accuracy, 4)}
        avg_loss = total_loss / total if total > 0 else 0.0
        # 统一打印前10条预测和真实token
        if debug_print_samples:
            print("[EVAL DEBUG] 前10条样本预测与真实token:")
            for idx, (pred_word, pred_token, label_word, label_token) in enumerate(debug_print_samples):
                print(f"[EVAL DEBUG] 样本{idx+1}: 预测='{pred_word}' (id:{pred_token}), 真实='{label_word}' (id:{label_token})")
        print(f"[Custom Eval] Loss: {avg_loss:.4f}  Accuracy: {accuracy:.4f}  (Total: {total})")
        self.model.train()
        return avg_loss, metrics

trainer = PrefixGenTrainer(
    model=model,
    args=my_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=decoder_tokenizer,
    data_collator=collate_fn
)

trainer.train()
trainer.save_model()
print(f"✓ 训练完成，模型已保存到 {args.output_dir}")

# 验证Prefix有效性
print("\n=== 验证Prefix有效性 ===")
model.eval()
for i, (prefix, target) in enumerate(prefix_target_pairs):
    input_ids = torch.tensor([i]).to(device)
    # 使用prefix embedding生成
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=None)
        logits = outputs.logits
        pred_token_id = logits[0, 0, :].argmax().item()
        pred_word = decoder_tokenizer.decode([pred_token_id])
        expected_token_id = gpt2_token_ids[target]
        print(f"前缀: '{prefix}' -> 预测: '{pred_word}' (id:{pred_token_id}) | 期望: '{target}' (id:{expected_token_id})")
print("=== 验证完成 ===")
