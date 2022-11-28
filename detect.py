from PIL import Image
from clip_interrogator import Interrogator, Config
image = Image.open("C:/Users/NakaMura/Desktop/2163670-bigthumbnail.jpg").convert('RGB')
ci = Interrogator(Config(clip_model_name="ViT-B-32/openai"))
print(ci.interrogate(image))