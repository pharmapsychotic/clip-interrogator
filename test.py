import sys
from PIL import Image
from clip_interrogator import Config, Interrogator
# import torch
image = Image.open('../pushd-gpt/output/resized.jpg').convert('RGB')
ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k", device='mps'))
print(ci.interrogate(image))
