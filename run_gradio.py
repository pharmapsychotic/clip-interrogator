#!/usr/bin/env python3
import argparse
import gradio as gr
import open_clip
from clip_interrogator import Interrogator, Config

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--share', action='store_true', help='Create a public link')
args = parser.parse_args()

ci = Interrogator(Config(cache_path="cache", clip_model_path="cache"))

def inference(image, mode, clip_model_name, blip_max_length, blip_num_beams):
    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()
    ci.config.blip_max_length = int(blip_max_length)
    ci.config.blip_num_beams = int(blip_num_beams)

    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)

models = ['/'.join(x) for x in open_clip.list_pretrained()]

inputs = [
    gr.inputs.Image(type='pil'),
    gr.Radio(['best', 'classic', 'fast'], label='Mode', value='best'),
    gr.Dropdown(models, value='ViT-L-14/openai', label='CLIP Model'),
    gr.Number(value=32, label='Caption Max Length'),
    gr.Number(value=64, label='Caption Num Beams'),
]
outputs = [
    gr.outputs.Textbox(label="Output"),
]

io = gr.Interface(
    inference, 
    inputs, 
    outputs, 
    title="🕵️‍♂️ CLIP Interrogator 🕵️‍♂️",
    allow_flagging=False,
)
io.launch(share=args.share)

