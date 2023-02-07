from clip_interrogator import Interrogator, Config
from fastapi import FastAPI
from PIL import Image
import requests
import torch

app = FastAPI()
images = {}


def inference(ci: Interrogator, image: Image, mode: str):
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)


@app.on_event("startup")
async def startup_event():
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU. Warning: this will be very slow!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = Config(device=device, clip_model_name='ViT-L-14/openai')
    ci = Interrogator(config)


@app.get("/image2prompt/{image_path}")
async def image2prompt(image_path: str):
    print(image_path)
    if str(image_path).startswith('http://') or str(image_path).startswith('https://'):
        image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    if not image:
        print(f'Error opening image {image_path}')
        exit(1)
    print(inference(ci, image, args.mode))

