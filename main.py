#!/usr/bin/env python3

import argparse
import clip
import requests
import torch

from PIL import Image

from clip_interrogator import CLIPInterrogator, Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='image file or url')
    parser.add_argument('-c', '--clip', default='ViT-L/14', help='name of CLIP model to use')

    args = parser.parse_args()
    if not args.image:
        parser.print_help()
        exit(1)

    # load image
    image_path = args.image
    if str(image_path).startswith('http://') or str(image_path).startswith('https://'):
        image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    if not image:
        print(f'Error opening image {image_path}')
        exit(1)

    # validate clip model name
    if args.clip not in clip.available_models():
        print(f"Could not find CLIP model {args.clip}!")
        print(f"    available models: {clip.available_models()}")
        exit(1)

    # generate a nice prompt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config(device=device, clip_model_name=args.clip, data_path='data')
    interrogator = CLIPInterrogator(config)
    prompt = interrogator.interrogate(image)
    print(prompt)

if __name__ == "__main__":
    main()
