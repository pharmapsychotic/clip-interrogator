from PIL import Image
#from clip_interrogator import Interrogator, Config
img = Image.open("C:/Users/NakaMura/Desktop/2163670-bigthumbnail.jpg").convert('RGB')
#ci = Interrogator(Config(clip_model_name="ViT-B-32/openai"))
#print(ci.interrogate(image))

import sys
sys.path.append('src/blip')
sys.path.append('clip-interrogator')

from clip_interrogator import Config, Interrogator

config = Config()
config.blip_num_beams = 64
config.blip_offload = False
config.chunk_size = 2048
config.flavor_intermediate_count = 2048

ci = Interrogator(config)

def inference(image, mode, clip_model_name, best_max_flavors=32):
    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)
        
print(inference(img, "fast", clip_model_name="ViT-B-32/openai"))