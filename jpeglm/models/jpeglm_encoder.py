#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JpegLM Encoder Model Architecture
将 JpegLM 从生成式语言模型改造成 encoder 架构进行分类任务
兼容 Hugging Face VisionEncoderDecoderModel 架构
"""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    GenerationMixin,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    EncoderDecoderModel,
    EncoderDecoderConfig,
)
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput, Seq2SeqLMOutput

class JpegLMEncoder(PreTrainedModel):
    """
    JpegLM Vision Encoder 用于 VisionEncoderDecoderModel
    专门设计为视觉编码器，输出与 VisionEncoderDecoderModel 兼容的格式
    """
    def __init__(self, config):
        super().__init__(config)
        
        # 加载预训练的 transformer 层（encoder 模式）
        self.model = AutoModel.from_pretrained(
            config.name_or_path,
            config=config
        )
        
        # 梯度检查点支持
        self.gradient_checkpointing = False
        
        # 执行初始化
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        self.model.set_input_embeddings(embeddings)

    def gradient_checkpointing_enable(self):
        """启用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        print("✓ JpegLMEncoder 已启用梯度检查点")

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        print("✓ JpegLMEncoder 已禁用梯度检查点")


    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播，返回 VisionEncoderDecoderModel 兼容的输出
        
        Args:
            input_ids: [batch_size, seq_len] - 也可能从 pixel_values 传递过来
            attention_mask: [batch_size, seq_len]
            
        Returns:
            BaseModelOutput: 包含 last_hidden_state 和 hidden_states
        """
        
        # 获取transformer输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )
        
        sequence_output = outputs.last_hidden_state
        
        # 对于 VisionEncoderDecoderModel，我们需要返回 BaseModelOutput
        # 并确保 last_hidden_state 的维度正确
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, 'attentions', None)
        )
        
class JpegLMEncoderWithPooling(JpegLMEncoder):
    """
    池化版 JpegLMEncoder，forward 直接输出池化特征
    支持 mean/max/cls/last 池化
    """
    def __init__(self, config):
        super().__init__(config)
        self.pooling_strategy = getattr(config, 'pooling_strategy', 'last')

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 获取 transformer 输出
        encoder_outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = encoder_outputs.last_hidden_state

        # attention_mask 转 float 并扩展维度
        if attention_mask is not None:
            mask = attention_mask.float()
        else:
            mask = torch.ones(sequence_output.shape[:2], device=sequence_output.device)

        # 池化
        if self.pooling_strategy == "mean":
            mask_expanded = mask.unsqueeze(-1).expand(sequence_output.size())
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            pooled_mask = (sum_mask > 0).long()  # [batch_size]
        elif self.pooling_strategy == "max":
            pooled_output = torch.max(sequence_output, dim=1)[0]
            pooled_mask = torch.ones(sequence_output.size(0), dtype=torch.long, device=sequence_output.device)  # [batch_size]
        elif self.pooling_strategy == "cls":
            pooled_output = sequence_output[:, 0]
            pooled_mask = mask[:, 0].long() if mask.dim() == 2 else torch.ones(sequence_output.size(0), dtype=torch.long, device=sequence_output.device)
        else:  # last
            seq_lengths = mask.sum(dim=1) - 1
            batch_size = sequence_output.size(0)
            pooled_output = sequence_output[torch.arange(batch_size), seq_lengths.long()]
            pooled_mask = torch.ones(batch_size, dtype=torch.long, device=sequence_output.device)  # [batch_size]

        # 输出规范化：last_hidden_state为池化特征，attention_mask与池化shape匹配，其余参数保持一致
        return BaseModelOutput(
            last_hidden_state=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=getattr(encoder_outputs, 'attentions', None),
        )
        
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel, shift_tokens_right
from transformers.utils import logging
import warnings

logger = logging.get_logger(__name__)

DEPRECATION_WARNING = (
    "Version v4.12.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.12.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

class JpegLMEncoderDecoderModelWithPooling(EncoderDecoderModel):
    """
    支持 JpegLMEncoderWithPooling 的 EncoderDecoderModel
    decoder 直接接受池化特征，自动创建投影层处理维度不一致
    """
    
    def _maybe_init_enc_to_dec_proj(self):
        """
        自动检测encoder/decoder hidden_size维度不一致时创建线性投影层
        """
        if not hasattr(self, 'enc_to_dec_proj') or self.enc_to_dec_proj is None:
            if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
            ):
                self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
            else:
                self.enc_to_dec_proj = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 自动初始化投影层（如有需要）
        self._maybe_init_enc_to_dec_proj()

        # 分离 encoder/decoder kwargs（与官方逻辑一致）
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)

        # 1. 获取 encoder 输出
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        # 2. 获取 encoder hidden states
        encoder_hidden_states = encoder_outputs[0]

        # 3. 池化逻辑：如果是池化输出（2D），转换为序列格式
        if encoder_hidden_states.dim() == 2:
            # 池化特征 [batch, hidden] -> [batch, 1, hidden]
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            # 池化后的 attention_mask：全1，因为只有一个 token
            pooled_attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=encoder_hidden_states.device)
        else:
            # 标准序列输出：保持原始 attention_mask
            pooled_attention_mask = attention_mask

        # 4. 线性投影（如有需要）- 与官方逻辑完全一致
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # 5. 处理 decoder_input_ids - 与官方逻辑完全一致
        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids.new_tensor(decoder_input_ids != self.config.pad_token_id)
        
        # 生成时如果 decoder_input_ids 为 None，自动初始化
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            batch_size = encoder_hidden_states.size(0)
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.config.decoder_start_token_id,
                dtype=torch.long,
                device=encoder_hidden_states.device
            )

        # 6. Decoder forward - 与官方逻辑完全一致
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=pooled_attention_mask,  # 使用池化处理后的 mask
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # 7. 损失计算 - 与官方逻辑完全一致
        loss = None
        if labels is not None:
            warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        # 8. 返回结果 - 与官方逻辑完全一致
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
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        """与官方方法保持一致"""
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        """与官方方法保持一致"""
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        """与官方方法保持一致"""
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, 
                                    encoder_outputs=None, **kwargs):
        """
        重写生成的输入准备方法，处理池化特征
        """
        # 如果 encoder_outputs 中的 last_hidden_state 是 2D（池化特征），
        # 需要转换为 3D 格式以兼容生成逻辑
        if encoder_outputs is not None:
            encoder_hidden_states = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]
            
            if encoder_hidden_states.dim() == 2:
                # 池化特征 [batch, hidden] -> [batch, 1, hidden]
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                # 创建新的 BaseModelOutput
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_hidden_states,
                    hidden_states=getattr(encoder_outputs, 'hidden_states', None),
                    attentions=getattr(encoder_outputs, 'attentions', None)
                )
                # 对应的 attention_mask 也需要调整
                if attention_mask is not None and attention_mask.shape[1] != 1:
                    attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=encoder_hidden_states.device)

        # 如果有 past_key_values，只保留最后一个 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        decoder_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache"),
        }
        
        return decoder_inputs

    def generate(self, input_ids=None, attention_mask=None, encoder_outputs=None, **kwargs):
        """
        移除自定义generate方法，直接使用父类的逻辑，问题已在forward中解决
        """
        # 如果没有提供 encoder_outputs，先运行 encoder
        if encoder_outputs is None and input_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

        # 处理池化特征
        if encoder_outputs is not None:
            encoder_hidden_states = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]
            if encoder_hidden_states.dim() == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_hidden_states,
                    hidden_states=getattr(encoder_outputs, 'hidden_states', None),
                    attentions=getattr(encoder_outputs, 'attentions', None)
                )
                if attention_mask is not None:
                    attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=encoder_hidden_states.device)

        # 调用父类生成方法，input_ids将在forward中自动初始化
        return super().generate(
            input_ids=None,  # 让forward方法自动处理
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **kwargs
        )


class JpegLMEncoderForClassification(JpegLMEncoder):
    """
    将 JpegLM 改造成 Encoder 架构的分类模型
    支持多种池化策略和自定义分类头
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_strategy = getattr(config, 'pooling_strategy', 'last')
        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.1)
        self.pre_classifier = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self._init_classifier_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 只负责池化和分类头，特征提取全部交给父类
        encoder_outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = encoder_outputs.last_hidden_state
        # 池化
        pooled_output = self._pool_hidden_states(sequence_output, attention_mask)
        # 分类头
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, 'hidden_states') else None,
            attentions=getattr(encoder_outputs, 'attentions', None)
        )
        
    def _init_classifier_weights(self):
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def _pool_hidden_states(self, sequence_output, attention_mask):
        if self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            pooled_output = torch.max(sequence_output, dim=1)[0]
        elif self.pooling_strategy == "cls":
            pooled_output = sequence_output[:, 0]
        else:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = sequence_output.size(0)
            pooled_output = sequence_output[torch.arange(batch_size), seq_lengths]
        return pooled_output

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel, GenerationMixin
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput

class JpegLMSeq2SeqModel(PreTrainedModel, GenerationMixin):
    def __init__(self, encoder_cfg, decoder_cfg, pooling='last'):
        cfg = encoder_cfg
        cfg.decoder_config = decoder_cfg
        cfg.encoder_pooling_strategy = pooling
        super().__init__(cfg)
        # 编码器
        encoder_cfg.is_decoder = False
        encoder_cfg.add_cross_attention = False
        self.encoder = AutoModel.from_pretrained(encoder_cfg.name_or_path, config=encoder_cfg)
        # 解码器
        decoder_cfg.is_decoder = True
        decoder_cfg.add_cross_attention = True
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_cfg.name_or_path, config=decoder_cfg)
        
        # 设置特殊token IDs - 确保它们被正确定义
        self.config.decoder_start_token_id = getattr(decoder_cfg, 'bos_token_id', None) or getattr(decoder_cfg, 'decoder_start_token_id', None)
        self.config.pad_token_id = getattr(decoder_cfg, 'pad_token_id', None) or getattr(decoder_cfg, 'eos_token_id', None)
        
        # 如果仍然没有pad_token_id，使用eos_token_id作为fallback
        if self.config.pad_token_id is None:
            self.config.pad_token_id = getattr(decoder_cfg, 'eos_token_id', 50256)  # GPT-2的eos_token_id
        
        # 如果没有decoder_start_token_id，使用bos_token_id或eos_token_id
        if self.config.decoder_start_token_id is None:
            self.config.decoder_start_token_id = self.config.pad_token_id

        # 投影层：如果 encoder/decoder hidden_size 不一致，则添加线性投影
        encoder_hidden_size = encoder_cfg.hidden_size
        decoder_hidden_size = decoder_cfg.hidden_size
        if encoder_hidden_size != decoder_hidden_size:
            self.encoder_to_decoder_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.encoder_to_decoder_proj = None
        
    def get_input_embeddings(self):
        # 对于 Seq2Seq 模型，返回编码器的嵌入层（处理输入图像）
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        # 设置编码器的嵌入层
        self.encoder.set_input_embeddings(embeddings)
        
    def get_output_embeddings(self):
        # 返回解码器的输出嵌入层（用于文本生成）
        return self.decoder.get_output_embeddings()
        
    def set_output_embeddings(self, embeddings):
        # 设置解码器的输出嵌入层
        self.decoder.set_output_embeddings(embeddings)
        
    def get_decoder_input_embeddings(self):
        # 返回解码器的输入嵌入层（处理文本输入）
        return self.decoder.get_input_embeddings()
        
    def set_decoder_input_embeddings(self, embeddings):
        # 设置解码器的输入嵌入层
        self.decoder.set_input_embeddings(embeddings)

    def forward(self, input_ids, attention_mask,
                decoder_input_ids=None, decoder_attention_mask=None,
                labels=None, **kwargs):
        # 编码
        enc_out = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               return_dict=True)
        hs = enc_out.last_hidden_state
        # 投影到 decoder hidden_size（如果需要）
        if self.encoder_to_decoder_proj is not None:
            hs = self.encoder_to_decoder_proj(hs)
        # 训练时，labels 中含 -100，需要先恢复再移位
        if decoder_input_ids is None and labels is not None:
            # 确保 labels 是正确的 tensor，避免梯度断开
            if not torch.is_tensor(labels):
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=hs.device)
            else:
                labels_tensor = labels.to(dtype=torch.long, device=hs.device)
            
            # 重要修复：先将 -100 替换为 pad_token_id，再进行 shift
            # 这样 shift_tokens_right 就能正确处理
            labels_for_shift = labels_tensor.clone()
            labels_for_shift[labels_for_shift == -100] = self.config.pad_token_id
            
            # 修正导入路径
            from transformers.models.bart.modeling_bart import shift_tokens_right
            decoder_input_ids = shift_tokens_right(
                labels_for_shift,  # 使用处理过的labels
                pad_token_id=self.config.pad_token_id,
                decoder_start_token_id=self.config.decoder_start_token_id
            )
        dec_out = self.decoder(input_ids=decoder_input_ids,
                               encoder_hidden_states=hs,
                               encoder_attention_mask=attention_mask,
                               return_dict=True)
        loss = None
        if labels is not None:
            logits = dec_out.logits
            # 确保 labels 是 LongTensor 且与 logits 在同一 device
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
            else:
                labels = labels.to(dtype=torch.long, device=logits.device)
            
            # 防止空 tensor
            if labels.numel() == 0:
                loss = logits.sum() * 0.0
            else:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # 检查损失是否为 NaN 或无穷大，但不要破坏梯度图
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ 异常损失值: {loss.item()}")
                    # 创建一个可微分的零损失，保持计算图
                    loss = (logits * 0.0).sum()
                elif loss.item() == 0.0:
                    print(f"⚠️ 损失为零，可能导致梯度消失")
                
                # 不要手动设置requires_grad，这会破坏计算图
        return Seq2SeqLMOutput(loss=loss,
                               logits=dec_out.logits,
                               past_key_values=dec_out.past_key_values,
                               decoder_hidden_states=dec_out.hidden_states,
                               cross_attentions=dec_out.cross_attentions,
                               encoder_last_hidden_state=hs)


def create_jpeglm_encoder(model_name_or_path, pooling_strategy='last', **kwargs):
    print(f"正在加载 JpegLM 配置: {model_name_or_path}")
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.is_decoder = False
    config.add_cross_attention = False
    config.use_cache = False
    config.pooling_strategy = pooling_strategy
    config.name_or_path = model_name_or_path
    
    print(f"正在加载 JpegLM 模型权重...")
    model = JpegLMEncoder(config)
    
    print(f"✓ 已创建 JpegLM Encoder")
    print(f"  - 池化策略: {pooling_strategy}")
    print(f"  - 隐藏维度: {config.hidden_size}")
    

    return model

def create_jpeglm_encoder_with_pooling(model_name_or_path, pooling_strategy='last', **kwargs):
    """
    创建池化版 JpegLMEncoder，forward 直接输出池化特征
    Args:
        model_name_or_path: 预训练模型路径
        pooling_strategy: 池化策略 ('mean', 'max', 'cls', 'last')
        **kwargs: 其他配置参数
    Returns:
        JpegLMEncoderWithPooling: 池化特征输出模型
    """
    print(f"正在加载 JpegLM 池化版配置: {model_name_or_path}")
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.is_decoder = False
    config.add_cross_attention = False
    config.use_cache = False
    config.pooling_strategy = pooling_strategy
    config.name_or_path = model_name_or_path
    model = JpegLMEncoderWithPooling(config)
    print(f"✓ 已创建 JpegLM 池化版 Encoder")
    print(f"  - 池化策略: {pooling_strategy}")
    print(f"  - 隐藏维度: {config.hidden_size}")
    return model

def create_jpeglm_encoder_cls_model(model_name_or_path, num_labels=10, pooling_strategy='last', **kwargs):
    """
    创建 JpegLM Encoder 分类模型
    
    Args:
        model_name_or_path: 预训练模型路径
        num_labels: 分类类别数
        pooling_strategy: 池化策略 ('mean', 'max', 'cls', 'last')
        **kwargs: 其他配置参数
        
    Returns:
        JpegLMEncoderForClassification: 配置好的分类模型
    """
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = num_labels
    config.use_cache = False
    config.is_decoder = False
    config.add_cross_attention = False
    config.pooling_strategy = pooling_strategy
    config.name_or_path = model_name_or_path
    
    # 可选配置
    if 'classifier_dropout' in kwargs:
        config.classifier_dropout = kwargs['classifier_dropout']
    # 直接创建JpegLMEncoderForClassification（继承自JpegLMEncoder）
    model = JpegLMEncoderForClassification(config)
    print(f"✓ 已创建 JpegLM Encoder 分类模型")
    print(f"  - 类别数: {num_labels}")
    print(f"  - 池化策略: {pooling_strategy}")
    print(f"  - 隐藏维度: {config.hidden_size}")
    return model

def create_seq2seq_model(
    encoder_model_name_or_path, 
    decoder_model_name_or_path,
    encoder_pooling_strategy='last',
    **kwargs
):
    """
    创建基于 JpegLM 的 Seq2Seq 模型，使用 Hugging Face EncoderDecoderModel
    
    Args:
        encoder_model_name_or_path: JpegLM 编码器模型路径
        decoder_model_name_or_path: 解码器模型路径 (如 GPT-2, T5等)
        encoder_pooling_strategy: 编码器池化策略
        **kwargs: 其他配置参数
        
    Returns:
        EncoderDecoderModel: 完整的 Seq2Seq 模型
    """
    print(f"正在创建基于 EncoderDecoderModel 的 Seq2Seq 模型...")
    
    # 创建编码器
    print(f"正在加载 JpegLM 编码器: {encoder_model_name_or_path}")
    encoder = create_jpeglm_encoder(
        encoder_model_name_or_path, 
        pooling_strategy=encoder_pooling_strategy
    )
    
    # 加载解码器
    print(f"正在加载解码器: {decoder_model_name_or_path}")
    decoder_config = AutoConfig.from_pretrained(decoder_model_name_or_path)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    
    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_model_name_or_path, 
        config=decoder_config
    )
    
    # 创建 EncoderDecoderConfig
    encoder_config = AutoConfig.from_pretrained(encoder_model_name_or_path)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder_config.use_cache = False
    encoder_config.pooling_strategy = encoder_pooling_strategy
    
    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )
    
    # 设置特殊token
    config.decoder_start_token_id = getattr(decoder_config, 'bos_token_id', None) or getattr(decoder_config, 'eos_token_id', 50256)
    config.pad_token_id = getattr(decoder_config, 'pad_token_id', None) or getattr(decoder_config, 'eos_token_id', 50256)
    config.eos_token_id = getattr(decoder_config, 'eos_token_id', 50256)
    
    # 创建自定义的 EncoderDecoderModel 子类，支持 input_ids
    class JpegLMEncoderDecoderModel(EncoderDecoderModel):
        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
    
    # 创建模型
    model = JpegLMEncoderDecoderModel(
        encoder=encoder,
        decoder=decoder,
        config=config
    )
    
    print(f"✓ 已创建 EncoderDecoderModel")
    print(f"  - 编码器: JpegLM ({encoder_pooling_strategy} pooling)")
    print(f"  - 解码器: {decoder_model_name_or_path}")
    print(f"  - 编码器维度: {encoder_config.hidden_size}")
    print(f"  - 解码器维度: {decoder_config.hidden_size}")
    
    return model


def create_seq2seq_model_legacy(
    encoder_model_name_or_path, 
    decoder_model_name_or_path,
    encoder_pooling_strategy='last',
    **kwargs
):
    """
    创建基于 JpegLM 的 Seq2Seq 模型（自定义实现，备用选项）
    
    Args:
        encoder_model_name_or_path: JpegLM 编码器模型路径
        decoder_model_name_or_path: 解码器模型路径 (如 GPT-2, T5等)
        encoder_pooling_strategy: 编码器池化策略
        **kwargs: 其他配置参数
        
    Returns:
        JpegLMSeq2SeqModel: 完整的 Seq2Seq 模型
    """
    print(f"正在加载 JpegLM 编码器配置: {encoder_model_name_or_path}")
    encoder_config = AutoConfig.from_pretrained(encoder_model_name_or_path)
    encoder_config.name_or_path = encoder_model_name_or_path
    
    print(f"正在加载解码器配置: {decoder_model_name_or_path}")
    decoder_config = AutoConfig.from_pretrained(decoder_model_name_or_path)
    decoder_config.name_or_path = decoder_model_name_or_path
    
    print(f"正在创建 Seq2Seq 模型...")
    model = JpegLMSeq2SeqModel(
        encoder_config,
        decoder_config,
        encoder_pooling_strategy
    )
    
    print(f"✓ 已创建 Seq2Seq 模型")
    print(f"  - 编码器: JpegLM ({encoder_pooling_strategy} pooling)")
    print(f"  - 解码器: {decoder_model_name_or_path}")
    print(f"  - 编码器维度: {encoder_config.hidden_size}")
    print(f"  - 解码器维度: {decoder_config.hidden_size}")
    
    return model


def load_jpeglm_encoder_model(model_path, **kwargs):
    """
    加载已训练的 JpegLM Encoder 模型
    
    Args:
        model_path: 模型保存路径
        **kwargs: 其他参数
        
    Returns:
        JpegLMEncoderForClassification: 加载的模型
    """
    try:
        # 尝试加载配置
        config = AutoConfig.from_pretrained(model_path)
        model = JpegLMEncoderForClassification.from_pretrained(model_path, config=config)
        print(f"✓ 已从 {model_path} 加载 JpegLM Encoder 模型")
        return model
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        raise e

