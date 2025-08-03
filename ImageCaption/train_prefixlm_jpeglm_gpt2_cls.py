# 示例运行命令：
# python /root/autodl-tmp/MLLM/ImageCaption/train_prefixlm_jpeglm_gpt2_cls.py --train_batch_size 2 --eval_batch_size 2 --eval_strategy steps --eval_steps 5 --logging_steps 64 --save_steps 512 --warmup_steps 512 --learning_rate 2e-4 --num_train_epochs 3 --save_total_limit 6 --lr_scheduler_type linear --gradient_accumulation_steps 8 --report_to none --bf16 --max_length 1024 --image_size 96 --num_train_samples 6000 --num_eval_samples 16

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, GenerationConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from jpeglm.models.jpeglm_encoder import create_jpeglm_encoder_with_pooling
from utils.data_utils import convert_img_to_bytes, create_preprocess_transform
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/MLLM/checkpoints/prefixlm-jpeglm-gpt2-mnist-classification")
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--image_size', type=int, default=28)
parser.add_argument('--max_length', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--num_train_samples', type=int, default=6000)
parser.add_argument('--num_eval_samples', type=int, default=1000)
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

# MNIST数字标签映射到文本
digit_to_text = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}
tiny_vocab = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
word2id = {w: i for i, w in enumerate(tiny_vocab)}
id2word = {i: w for i, w in enumerate(tiny_vocab)}

# 加载tokenizer和模型
encoder_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/MLLM/models/jpeg-lm")
decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")
decoder_tokenizer.pad_token = decoder_tokenizer.unk_token

# 将tiny_vocab中的单词转换为GPT2 tokenizer中的token IDs
gpt2_token_ids = {}
for word in tiny_vocab:
    token_ids = decoder_tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) == 1:
        gpt2_token_ids[word] = token_ids[0]
    else:
        print(f"警告: '{word}' 被tokenize为多个token: {token_ids}")
        gpt2_token_ids[word] = token_ids[0]  # 取第一个token

print("数字词汇到GPT2 token ID的映射:")
for word, token_id in gpt2_token_ids.items():
    decoded = decoder_tokenizer.decode([token_id])
    print(f"  {word} -> token_id: {token_id}, decoded: '{decoded}'")

gpt2_config = AutoConfig.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 加载JpegLM池化encoder
jpeglm_encoder = create_jpeglm_encoder_with_pooling("/root/autodl-tmp/MLLM/models/jpeg-lm", pooling_strategy='last').to(device)

# 分类投影层（将池化特征投影到gpt2的hidden_size）
proj = torch.nn.Linear(jpeglm_encoder.config.hidden_size, gpt2_config.n_embd).to(device)

# PrefixLM模型封装
class PrefixLMForClassification(PreTrainedModel, GenerationMixin):
    supports_gradient_checkpointing = True
    main_input_name = "input_ids"
    base_model_prefix = "prefix_lm"

    def __init__(self, encoder, proj, decoder, decoder_tokenizer, bos_token_id):
        # 使用decoder的config初始化PreTrainedModel
        super().__init__(decoder.config)
        self.encoder = encoder
        self.decoder = decoder
        self.proj = proj
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # 投影encoder输出到decoder维度
        encoder_hidden_states = self.proj(encoder_hidden_states)
        # 取池化后的特征作为prefix，reshape为[batch, 1, hidden]
        prefix_embeds = encoder_hidden_states.unsqueeze(1)  # [batch, 1, hidden]

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
            if decoder_attention_mask is None and decoder_input_ids is not None:
                decoder_attention_mask = decoder_input_ids.new_tensor(decoder_input_ids != self.config.pad_token_id)

        # 如果有decoder_inputs_embeds，与prefix_embeds拼接
        if decoder_inputs_embeds is not None:
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
            **kwargs_decoder,
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
            loss_fct = CrossEntropyLoss()
            # 对于分类任务，只计算第一个位置的loss
            if logits.dim() == 3:  # [batch, seq_len, vocab_size]
                first_token_logits = logits[:, 0, :]  # [batch, vocab_size]
                loss = loss_fct(first_token_logits, labels.view(-1))
            else:
                loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=getattr(decoder_outputs, 'cross_attentions', None),
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def get_input_embeddings(self):
        # 对encoder做LoRA时，返回encoder的embedding层
        if hasattr(self.encoder, 'get_input_embeddings'):
            return self.encoder.get_input_embeddings()
        elif hasattr(self.encoder, 'embeddings'):
            return self.encoder.embeddings
        else:
            return None

    def set_input_embeddings(self, value):
        if hasattr(self.encoder, 'set_input_embeddings'):
            self.encoder.set_input_embeddings(value)
        elif hasattr(self.encoder, 'embeddings'):
            self.encoder.embeddings = value

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 对于分类任务，labels是[batch_size]的一维张量，需要转换为[batch_size, 1]
        if labels is None:
            return None
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        
        # 确保config参数存在
        pad_token_id = getattr(self.config, 'pad_token_id', self.decoder_tokenizer.pad_token_id or self.decoder_tokenizer.unk_token_id)
        decoder_start_token_id = getattr(self.config, 'decoder_start_token_id', self.bos_token_id)
        
        return shift_tokens_right(labels, pad_token_id, decoder_start_token_id)

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, **model_kwargs):
        # 兼容transformers/PEFT生成流程，直接 passthrough
        return model_kwargs
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # 兼容transformers/PEFT生成流程，直接 passthrough
        return {"input_ids": input_ids, **kwargs}

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the PrefixLMForClassification directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

# 数据集
class MNISTJpegBytesPrefixDataset(Dataset):
    def __init__(self, hf_dataset, encoder_tokenizer, decoder_tokenizer, digit_to_text, max_length=1024, image_size=28):
        self.dataset = hf_dataset
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.digit_to_text = digit_to_text
        self.max_length = max_length
        self.transform = create_preprocess_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = item['label']
        caption = self.digit_to_text[label]
        img = item['image'].convert("RGB")
        img = self.transform(img)
        jpeg_str = convert_img_to_bytes(img)
        input_ids = [self.encoder_tokenizer.bos_token_id] + self.encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
        input_ids = input_ids[:self.max_length]
        # 使用GPT2 tokenizer中的真实token ID
        caption = self.digit_to_text[label]
        label_token_id = gpt2_token_ids[caption]
        return {"input_ids": torch.tensor(input_ids), "label_id": torch.tensor(label_token_id)}

# 加载MNIST数据集
mnist_dataset = load_dataset("ylecun/mnist")
train_data = mnist_dataset["train"].select(range(min(args.num_train_samples, len(mnist_dataset["train"]))))
test_data = mnist_dataset["test"].select(range(min(args.num_eval_samples, len(mnist_dataset["test"]))))

train_dataset = MNISTJpegBytesPrefixDataset(train_data, encoder_tokenizer, decoder_tokenizer, digit_to_text, args.max_length, args.image_size)
val_dataset = MNISTJpegBytesPrefixDataset(test_data, encoder_tokenizer, decoder_tokenizer, digit_to_text, args.max_length, args.image_size)

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    label_ids = [item["label_id"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=encoder_tokenizer.pad_token_id)
    attention_mask = (input_ids_padded != encoder_tokenizer.pad_token_id).long()
    label_ids = torch.stack(label_ids)
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask, "labels": label_ids}

# MySeq2SeqTrainer集成
from hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments
bos_token_id = decoder_tokenizer.bos_token_id or decoder_tokenizer.eos_token_id or 50256

# 在模型外部确保特殊token id全部设置到config
model = PrefixLMForClassification(jpeglm_encoder, proj, gpt2_model, decoder_tokenizer, bos_token_id)
model.config.decoder_start_token_id = bos_token_id
model.config.pad_token_id = decoder_tokenizer.pad_token_id or decoder_tokenizer.unk_token_id

def compute_metrics(pred):
    labels = pred.label_ids  # [batch_size] - 包含GPT2 token IDs
    preds = pred.predictions # [batch_size, seq_len, vocab_size] 或 [batch_size, seq_len]
    
    # 处理预测结果：如果是logits，取argmax；如果已经是token ids，直接使用
    if len(preds.shape) == 3:  # logits: [batch_size, seq_len, vocab_size]
        pred_token_ids = np.argmax(preds, axis=-1)  # [batch_size, seq_len]
    else:  # token ids: [batch_size, seq_len]
        pred_token_ids = preds
    
    # 取第1个位置的预测（现在序列长度为1）
    if pred_token_ids.shape[1] > 0:
        pred_tokens = pred_token_ids[:, 0]  # [batch_size]
    else:
        pred_tokens = np.zeros_like(labels)  # fallback
    
    # 计算准确率 - 比较GPT2 token IDs
    correct = (pred_tokens == labels).sum()
    total = len(labels)
    accuracy = correct / total if total > 0 else 0.0
    
    # 调试信息：打印前几个预测和标签
    if len(pred_tokens) > 0:
        print(f"[DEBUG] 前3个预测token IDs: {pred_tokens[:3]}")
        print(f"[DEBUG] 前3个真实token IDs: {labels[:3]}")
        for i in range(min(3, len(pred_tokens))):
            pred_word = decoder_tokenizer.decode([pred_tokens[i]])
            label_word = decoder_tokenizer.decode([labels[i]])
            print(f"[DEBUG] 样本{i}: 预测='{pred_word}' (id:{pred_tokens[i]}), 真实='{label_word}' (id:{labels[i]})")
    
    return {"accuracy": round(float(accuracy), 4)}

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
    

# LoRA配置

# 获取GPT2所有关键模块名称，作为modules_to_save
h_modules = [f"decoder.transformer.h.{i}" for i in range(model.decoder.config.n_layer)]
gpt2_modules = h_modules + [
    "decoder.transformer.ln_f",
    "decoder.lm_head",
]

# 添加其他需要保存的模块
modules_to_save = gpt2_modules + [
    # "proj",  # 投影层
]

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    target_modules=[
        # Encoder中的线性层 (JpegLM相关)
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention层
        "gate_proj", "up_proj", "down_proj",     # MLP层  
    ],
    modules_to_save=modules_to_save,  # 保存GPT2全部模块和投影层
    lora_dropout=0.1,
)

model.encoder.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)

# 打印LoRA训练参数统计
print("\n==== LoRA训练参数统计 ====")
model.print_trainable_parameters()

# 详细打印各层参数状态
print("\n==== 各层参数 requires_grad 状态 ====")
trainable_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"{name:80} requires_grad={param.requires_grad} ({param.numel():,} params)")

print(f"\n总训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
print("==== END ====")

trainer = MySeq2SeqTrainer(
    model=model,
    args=my_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=decoder_tokenizer,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

trainer.train()
trainer.save_model()
print(f"✓ 训练完成，模型已保存到 {args.output_dir}")
