from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from PIL import Image
import datasets
import os

# ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image-captioning-output", "checkpoint-1000")
ckpt_dir = "NourFakih/image-captioning-Vit-GPT2-Flickr8k"
model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

ds_eval = datasets.load_dataset("jxie/flickr8k", split='validation')
for i in range(5):
    image = ds_eval[i]['image']  # 获取第一张图片
    image.save(f'/root/autodl-tmp/MLLM/ImageCaption/images/example_{i}.jpeg')  # 保存图片以便查看
    origin_caption = ds_eval[i]['caption_0']  # 获取对应的caption
    # image = Image.open('/root/autodl-tmp/MLLM/datasets/flickr8k/images/sample_3140_example.jpg')
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    output_ids = model.generate(pixel_values)
    gen_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("" + "="*50)
    print(f"generatd caption: {gen_caption}\noriginal caption: {origin_caption}")