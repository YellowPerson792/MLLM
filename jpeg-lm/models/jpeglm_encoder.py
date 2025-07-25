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
)
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput, Seq2SeqLMOutput


class JpegLMEncoderForClassification(PreTrainedModel):
    """
    将 JpegLM 改造成 Encoder 架构的分类模型
    支持多种池化策略和自定义分类头
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # 加载预训练的 transformer 层（encoder 模式）
        self.model = AutoModel.from_pretrained(
            config.name_or_path,
            config=config
        )
        
        # 分类头组件
        self.dropout = nn.Dropout(config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.3)
        self.pre_classifier = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 池化策略
        self.pooling_strategy = getattr(config, 'pooling_strategy', 'last')
        
        # 执行初始化
        self.post_init()
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """自定义初始化分类头权重，确保初始 logits 接近 0"""
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        self.model.set_input_embeddings(embeddings)

    def _pool_hidden_states(self, sequence_output, attention_mask):
        """
        根据策略池化隐藏状态
        
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        if self.pooling_strategy == "mean":
            # 平均池化（排除padding）
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            
        elif self.pooling_strategy == "max":
            # 最大池化
            pooled_output = torch.max(sequence_output, dim=1)[0]
            
        elif self.pooling_strategy == "cls":
            # 使用第一个token ([CLS] 风格)
            pooled_output = sequence_output[:, 0]
            
        else:  # last (default for GPT-style models)
            # 使用最后一个非padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = sequence_output.size(0)
            pooled_output = sequence_output[torch.arange(batch_size), seq_lengths]
            
        return pooled_output

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] (可选，用于计算损失)
            
        Returns:
            dict: 包含 loss (如果提供labels) 和 logits
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
        
        # 池化
        pooled_output = self._pool_hidden_states(sequence_output, attention_mask)
        
        # 分类头
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 使用 reduction='mean' 确保损失正确平均，适配梯度累积
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if outputs.hidden_states else None,
            'last_hidden_state': sequence_output
        }


class JpegLMVisionEncoder(PreTrainedModel):
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
        
        # 池化策略
        self.pooling_strategy = getattr(config, 'pooling_strategy', 'last')
        
        # 执行初始化
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        self.model.set_input_embeddings(embeddings)

    def _pool_hidden_states(self, sequence_output, attention_mask):
        """
        根据策略池化隐藏状态
        
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        if self.pooling_strategy == "mean":
            # 平均池化（排除padding）
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            
        elif self.pooling_strategy == "max":
            # 最大池化
            pooled_output = torch.max(sequence_output, dim=1)[0]
            
        elif self.pooling_strategy == "cls":
            # 使用第一个token ([CLS] 风格)
            pooled_output = sequence_output[:, 0]
            
        else:  # last (default for GPT-style models)
            # 使用最后一个非padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = sequence_output.size(0)
            pooled_output = sequence_output[torch.arange(batch_size), seq_lengths]
            
        return pooled_output

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播，返回 VisionEncoderDecoderModel 兼容的输出
        
        Args:
            input_ids: [batch_size, seq_len] - 也可能从 pixel_values 传递过来
            attention_mask: [batch_size, seq_len]
            
        Returns:
            BaseModelOutput: 包含 last_hidden_state 和 hidden_states
        """
        # VisionEncoderDecoderModel 可能会以 pixel_values 的名字传递 input_ids
        if input_ids is None and 'pixel_values' in kwargs:
            input_ids = kwargs.pop('pixel_values')
        
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


class JpegLMSeq2SeqModel(PreTrainedModel, GenerationMixin):
    """
    基于 JpegLM 编码器和 GPT-2 解码器的 Seq2Seq 模型
    专门用于图像描述任务，支持长序列输入，避免 VisionEncoderDecoderModel 的限制
    """
    
    def __init__(self, encoder_config, decoder_config, encoder_pooling_strategy='last'):
        # 创建一个合并的配置
        config = encoder_config
        config.decoder_config = decoder_config
        config.encoder_pooling_strategy = encoder_pooling_strategy
        
        super().__init__(config)
        
        # 编码器：JpegLM (支持长序列)
        encoder_config.is_decoder = False
        encoder_config.add_cross_attention = False
        encoder_config.use_cache = False
        encoder_config.pooling_strategy = encoder_pooling_strategy
        
        self.encoder = AutoModel.from_pretrained(
            encoder_config.name_or_path,
            config=encoder_config
        )
        
        # 解码器：GPT-2 (配置为解码器模式)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.use_cache = True
        
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_config.name_or_path,
            config=decoder_config
        )
        
        # 编码器到解码器的投影层
        if encoder_config.hidden_size != decoder_config.hidden_size:
            self.encoder_decoder_proj = nn.Linear(
                encoder_config.hidden_size, 
                decoder_config.hidden_size
            )
        else:
            self.encoder_decoder_proj = None
            
        # 池化策略
        self.pooling_strategy = encoder_pooling_strategy
        
        # 设置生成配置
        self.config.decoder_start_token_id = decoder_config.bos_token_id
        self.config.pad_token_id = decoder_config.pad_token_id
        self.config.eos_token_id = decoder_config.eos_token_id
        
    def get_encoder(self):
        return self.encoder
        
    def get_decoder(self):
        return self.decoder
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 编码器前向传播
        if encoder_outputs is None:
            # 确保输入序列长度不会导致位置编码问题
            if input_ids is not None:
                seq_len = input_ids.size(1)
                # 检查序列长度是否超过模型的最大位置编码
                max_pos = getattr(self.encoder.config, 'max_position_embeddings', 2048)
                if seq_len > max_pos:
                    print(f"警告: 序列长度 {seq_len} 超过最大位置编码 {max_pos}，截断到 {max_pos}")
                    input_ids = input_ids[:, :max_pos]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :max_pos]
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                use_cache=False  # 编码器不需要缓存
            )
            
        # 获取编码器隐藏状态
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # 投影到解码器维度（如果需要）
        if self.encoder_decoder_proj is not None:
            encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)
            
        # 准备解码器输入
        if decoder_input_ids is None and labels is not None:
            # 训练时，使用 labels 作为解码器输入（teacher forcing）
            decoder_input_ids = self._shift_right(labels)
            
        # 解码器前向传播
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # 计算损失
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        if not return_dict:
            outputs = (decoder_outputs.logits,) + decoder_outputs[1:]
            return (loss,) + outputs if loss is not None else outputs
            
        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=getattr(decoder_outputs, 'cross_attentions', None),
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=getattr(encoder_outputs, 'attentions', None),
        )
        
    def _shift_right(self, input_ids):
        """向右移位，用于 teacher forcing"""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.config.decoder_start_token_id
        return shifted_input_ids


def create_jpeglm_vision_encoder(model_name_or_path, pooling_strategy='last', **kwargs):
    """
    创建 JpegLM Vision Encoder 用于 VisionEncoderDecoderModel
    
    Args:
        model_name_or_path: 预训练模型路径
        pooling_strategy: 池化策略 ('mean', 'max', 'cls', 'last')
        **kwargs: 其他配置参数
        
    Returns:
        JpegLMVisionEncoder: 配置好的视觉编码器
    """
    print(f"正在加载 JpegLM 配置: {model_name_or_path}")
    config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Encoder 模式配置
    config.is_decoder = False
    config.add_cross_attention = False
    config.use_cache = False
    
    # Vision Encoder 相关配置
    config.pooling_strategy = pooling_strategy
    config.name_or_path = model_name_or_path
    
    print(f"正在加载 JpegLM 模型权重 (13GB)，请耐心等待...")
    model = JpegLMVisionEncoder(config)
    
    print(f"✓ 已创建 JpegLM Vision Encoder")
    print(f"  - 池化策略: {pooling_strategy}")
    print(f"  - 隐藏维度: {config.hidden_size}")
    
    return model


def create_vision_encoder_decoder_model(
    encoder_model_name_or_path, 
    decoder_model_name_or_path,
    encoder_pooling_strategy='last',
    **kwargs
):
    """
    创建基于 JpegLM 的 VisionEncoderDecoderModel
    
    Args:
        encoder_model_name_or_path: JpegLM 编码器模型路径
        decoder_model_name_or_path: 解码器模型路径 (如 GPT-2, T5等)
        encoder_pooling_strategy: 编码器池化策略
        **kwargs: 其他配置参数
        
    Returns:
        VisionEncoderDecoderModel: 完整的视觉编码-解码模型
    """
    # 创建编码器
    encoder = create_jpeglm_vision_encoder(
        encoder_model_name_or_path, 
        pooling_strategy=encoder_pooling_strategy
    )
    
    print(f"正在从缓存加载解码器: {decoder_model_name_or_path}")
    
    # 使用缓存加载解码器，优先使用本地缓存
    from transformers import AutoModelForCausalLM, AutoConfig
    
    # 直接加载，Transformers 会自动使用缓存
    decoder_config = AutoConfig.from_pretrained(
        decoder_model_name_or_path,
        cache_dir='/root/.cache/huggingface'
    )
    
    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_model_name_or_path,
        config=decoder_config,
        cache_dir='/root/.cache/huggingface'
    )
    print(f"✓ 成功加载解码器: {decoder_model_name_or_path}")
    
    # 使用已加载的配置，避免重复加载
    print(f"正在创建 VisionEncoderDecoderConfig...")
    encoder_config = AutoConfig.from_pretrained(encoder_model_name_or_path)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False  
    encoder_config.use_cache = False
    encoder_config.pooling_strategy = encoder_pooling_strategy
    
    # 创建 VisionEncoderDecoderConfig，使用已有的 decoder_config
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )
    print(f"✓ 配置创建完成")
    
    # 使用 VisionEncoderDecoderModel 直接传入已加载的 encoder 和 decoder
    print(f"正在组装 VisionEncoderDecoderModel...")
    
    # 创建一个自定义的 VisionEncoderDecoderModel 子类，支持 input_ids
    class JpegLMVisionEncoderDecoderModel(VisionEncoderDecoderModel):
        def forward(
            self,
            pixel_values=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # 添加对 input_ids 的支持
            input_ids=None,
            attention_mask=None,
            **kwargs,
        ):
            # 如果提供了 input_ids，将其作为 pixel_values 使用
            if input_ids is not None and pixel_values is None:
                pixel_values = input_ids
            if attention_mask is not None and 'encoder_attention_mask' not in kwargs:
                kwargs['encoder_attention_mask'] = attention_mask
                
            return super().forward(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
    
    # 直接使用自定义的 VisionEncoderDecoderModel，传入已加载的模型实例
    model = JpegLMVisionEncoderDecoderModel(
        encoder=encoder,
        decoder=decoder,
        config=config
    )
    
    # 设置特殊token
    model.config.decoder_start_token_id = decoder.config.bos_token_id
    model.config.pad_token_id = decoder.config.pad_token_id
    
    print(f"✓ 已创建 VisionEncoderDecoderModel")
    print(f"  - 编码器: JpegLM ({encoder_pooling_strategy} pooling)")
    print(f"  - 解码器: {decoder_model_name_or_path}")
    
    return model


def create_seq2seq_model(
    encoder_model_name_or_path, 
    decoder_model_name_or_path,
    encoder_pooling_strategy='last',
    **kwargs
):
    """
    创建基于 JpegLM 的 Seq2Seq 模型，避免 VisionEncoderDecoderModel 的序列长度限制
    
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
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        encoder_pooling_strategy=encoder_pooling_strategy
    )
    
    print(f"✓ 已创建 Seq2Seq 模型")
    print(f"  - 编码器: JpegLM ({encoder_pooling_strategy} pooling)")
    print(f"  - 解码器: {decoder_model_name_or_path}")
    print(f"  - 编码器维度: {encoder_config.hidden_size}")
    print(f"  - 解码器维度: {decoder_config.hidden_size}")
    
    return model


def create_vision_encoder_decoder_config(
    encoder_model_name_or_path,
    decoder_model_name_or_path,
    encoder_pooling_strategy='last'
):
    """
    创建 VisionEncoderDecoderConfig
    
    Args:
        encoder_model_name_or_path: 编码器模型路径
        decoder_model_name_or_path: 解码器模型路径
        encoder_pooling_strategy: 编码器池化策略
        
    Returns:
        VisionEncoderDecoderConfig: 配置对象
    """
    # 加载编码器和解码器配置
    encoder_config = AutoConfig.from_pretrained(encoder_model_name_or_path)
    decoder_config = AutoConfig.from_pretrained(decoder_model_name_or_path)
    
    # 修改编码器配置
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder_config.use_cache = False
    encoder_config.pooling_strategy = encoder_pooling_strategy
    
    # 创建 VisionEncoderDecoderConfig
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )
    
    return config


def create_jpeglm_encoder_model(model_name_or_path, num_labels=10, pooling_strategy='last', **kwargs):
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
    
    # 基础配置
    config.num_labels = num_labels
    config.use_cache = False
    
    # Encoder 模式配置
    config.is_decoder = False
    config.add_cross_attention = False
    
    # 分类相关配置
    config.pooling_strategy = pooling_strategy
    config.name_or_path = model_name_or_path
    
    # 可选配置
    if 'classifier_dropout' in kwargs:
        config.classifier_dropout = kwargs['classifier_dropout']
    
    model = JpegLMEncoderForClassification(config)
    
    print(f"✓ 已创建 JpegLM Encoder 分类模型")
    print(f"  - 类别数: {num_labels}")
    print(f"  - 池化策略: {pooling_strategy}")
    print(f"  - 隐藏维度: {config.hidden_size}")
    
    return model
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
    
    # 基础配置
    config.num_labels = num_labels
    config.use_cache = False
    
    # Encoder 模式配置
    config.is_decoder = False
    config.add_cross_attention = False
    
    # 分类相关配置
    config.pooling_strategy = pooling_strategy
    config.name_or_path = model_name_or_path
    
    # 可选配置
    if 'classifier_dropout' in kwargs:
        config.classifier_dropout = kwargs['classifier_dropout']
    
    model = JpegLMEncoderForClassification(config)
    
    print(f"✓ 已创建 JpegLM Encoder 分类模型")
    print(f"  - 类别数: {num_labels}")
    print(f"  - 池化策略: {pooling_strategy}")
    print(f"  - 隐藏维度: {config.hidden_size}")
    
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


if __name__ == "__main__":
    # 测试模型创建
    print("=== JpegLM Encoder 模型测试 ===")
    
    # 1. 测试分类模型
    print("\n--- 测试分类模型 ---")
    # model = create_jpeglm_encoder_model(
    #     model_name_or_path="/path/to/jpeg-lm",
    #     num_labels=10,
    #     pooling_strategy='mean'
    # )
    
    # 2. 测试 Vision Encoder
    print("\n--- 测试 Vision Encoder ---")
    # vision_encoder = create_jpeglm_vision_encoder(
    #     model_name_or_path="/path/to/jpeg-lm",
    #     pooling_strategy='mean'
    # )
    
    # 3. 测试 VisionEncoderDecoderModel
    print("\n--- 测试 VisionEncoderDecoderModel ---")
    # vision_encoder_decoder = create_vision_encoder_decoder_model(
    #     encoder_model_name_or_path="/path/to/jpeg-lm",
    #     decoder_model_name_or_path="gpt2",  # 或者其他解码器
    #     encoder_pooling_strategy='mean'
    # )
    
    print("\n=== 使用示例 ===")
    print("# 1. 创建分类模型")
    print("from jpeglm_encoder import create_jpeglm_encoder_model")
    print("model = create_jpeglm_encoder_model('/path/to/jpeg-lm', num_labels=10)")
    print()
    print("# 2. 创建 Vision Encoder")
    print("from jpeglm_encoder import create_jpeglm_vision_encoder")
    print("encoder = create_jpeglm_vision_encoder('/path/to/jpeg-lm')")
    print()
    print("# 3. 创建 VisionEncoderDecoderModel")
    print("from jpeglm_encoder import create_vision_encoder_decoder_model")
    print("model = create_vision_encoder_decoder_model(")
    print("    encoder_model_name_or_path='/path/to/jpeg-lm',")
    print("    decoder_model_name_or_path='gpt2'")
    print(")")
    print()
    print("# 4. 使用 VisionEncoderDecoderModel 进行推理")
    print("# 输入: input_ids (图像token序列)")
    print("# 输出: 生成的文本序列")
    print("outputs = model.generate(")
    print("    input_ids=input_ids,")
    print("    attention_mask=attention_mask,")
    print("    max_length=50")
    print(")")
    
    print("\n模型结构定义完成！")
