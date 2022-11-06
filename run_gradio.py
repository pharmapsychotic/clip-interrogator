#!/usr/bin/env python3
import clip
import gradio as gr
from clip_interrogator import Interrogator, Config

ci = Interrogator(Config())

def inference(image, mode, clip_model_name, blip_max_length, blip_num_beams):
    global ci
    if clip_model_name != ci.config.clip_model_name:
        ci = Interrogator(Config(clip_model_name=clip_model_name))
    ci.config.blip_max_length = int(blip_max_length)
    ci.config.blip_num_beams = int(blip_num_beams)

    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)
    
inputs = [
    gr.inputs.Image(type='pil'),
    gr.Radio(['best', 'classic', 'fast'], label='Mode', value='best'),
    gr.Dropdown(clip.available_models(), value='ViT-L/14', label='CLIP Model'),
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
io.launch()
