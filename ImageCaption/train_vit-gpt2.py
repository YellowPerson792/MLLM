from transformers import GPT2Config, ViTConfig, VisionEncoderDecoderConfig,VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["WANDB_DISABLED"] = 'True'  # 禁用wandb

image_encoder_model = "google/vit-base-patch16-224-in21k"
text_decode_model = "gpt2"

# model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
#     image_encoder_model, text_decode_model
# )
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)

# 设置一个少见字符为pad_token，并与eos_token区分
if "[PAD]" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.decoder.resize_token_embeddings(len(tokenizer))

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

from transformers import default_data_collator

import evaluate
metric = evaluate.load("rouge")

import numpy as np

ignore_pad_token_for_loss = True

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # 用简单的'. '分句替代nltk分句
    def simple_sent_tokenize(text):
        return text.split('. ')

    preds = ['\n'.join(simple_sent_tokenize(pred)) for pred in preds]
    labels = ['\n'.join(simple_sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


import datasets
ds_train = datasets.load_dataset("jxie/flickr8k", split='train')
ds_eval = datasets.load_dataset("jxie/flickr8k", split='validation')

def tokenization_fn(captions, max_target_length):
    # 手动在caption末尾加eos_token，确保一定有eos
    if isinstance(captions, str):
        captions = captions + tokenizer.eos_token
    else:
        captions = [c + tokenizer.eos_token for c in captions]

    inputs = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_target_length,
        return_tensors="np",
        add_special_tokens=False  # 已经手动加了eos，不再自动加
    )
    
    input_ids = inputs["input_ids"]

    # Replace pad_token_id with -100 for loss masking
    labels = np.where(input_ids == tokenizer.pad_token_id, -100, input_ids)
    return labels

# image preprocessing step
def feature_extraction_fn(image_paths):

    encoder_inputs = feature_extractor(images=image_paths, return_tensors="np")

    return encoder_inputs.pixel_values

def preprocess_fn(examples, max_target_length):
    image_paths = examples['image'] if 'image' in examples else examples.get('filename', None)
    captions = examples['text'] if 'text' in examples else examples.get('caption_0', None)
    input_ids = tokenization_fn(captions, max_target_length)
    model_inputs = {
        'labels': input_ids,     # 训练时labels和input_ids一致
        'pixel_values': feature_extraction_fn(image_paths),
    }
    return model_inputs

processed_train_dataset = ds_train.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    remove_columns=ds_train.column_names,
    num_proc=12,  # 使用12个进程并行处理
    desc="处理数据集"
)

processed_eval_dataset = ds_eval.map(
    function=preprocess_fn,
    batched=True,
    fn_kwargs={"max_target_length": 128},
    remove_columns=ds_train.column_names,
    num_proc=12,  # 使用12个进程并行处理
    desc="处理数据集"
)


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="epoch",
    eval_steps=10,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    logging_steps=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "image-captioning-output"),
)

# flickr8k只有train split，eval_dataset设为None
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_eval_dataset,
    data_collator=default_data_collator,  # 用新的collator
)

# 查看前3个样本的详细内容
import numpy as np
for i in range(1):
    print(f"Sample {i}:")
    label_ids = processed_train_dataset[i]['labels']
    decoded = tokenizer.decode([t for t in label_ids if t != -100], skip_special_tokens=False)
    print("Decoded label:", decoded)
    # 转为numpy后再取shape
    pixel = processed_train_dataset[i]['pixel_values']
    pixel_np = np.array(pixel)
    print("Pixel shape:", pixel_np.shape)
    print("-" * 40)
    
# 冻结gpt2所有参数
for param in model.decoder.parameters():
    param.requires_grad = False

# 只微调gpt2最后一层和LayerNorm/Head
for name, param in model.decoder.named_parameters():
    if any([k in name for k in ["h.11", "ln_f", "lm_head"]]):  # gpt2-base最后一层是h.11
        param.requires_grad = True
        
for name, param in model.named_parameters():
    print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

trainer.train()
trainer.save_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "image-captioning-output"))