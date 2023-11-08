# clip-interrogator

*Want to figure out what a good prompt might be to create new images like an existing one? The **CLIP Interrogator** is here to get you answers!*

## Run it!

🆕 Now available as a [Stable Diffusion Web UI Extension](https://github.com/pharmapsychotic/clip-interrogator-ext)! 🆕

<br>

Run Version 2 on Colab, HuggingFace, Replicate, and Tungsten!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb) [![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/pharma/CLIP-Interrogator) [![Replicate](https://replicate.com/pharmapsychotic/clip-interrogator/badge)](https://replicate.com/pharmapsychotic/clip-interrogator) [![Tungsten](https://tungsten.run/mjpyeon/clip-interrogator/_badge)](https://tungsten.run/mjpyeon/clip-interrogator) [![Lambda](https://img.shields.io/badge/%CE%BB-Lambda-blue)](https://cloud.lambdalabs.com/demos/ml/CLIP-Interrogator)

<br>


Version 1 still available in Colab for comparing different CLIP models 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/v1/clip_interrogator.ipynb) 


## About

The **CLIP Interrogator** is a prompt engineering tool that combines OpenAI's [CLIP](https://openai.com/blog/clip/) and Salesforce's [BLIP](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/) to optimize text prompts to match a given image. Use the resulting prompts with text-to-image models like [Stable Diffusion](https://github.com/CompVis/stable-diffusion) on [DreamStudio](https://beta.dreamstudio.ai/) to create cool art!


## Using as a library

Create and activate a Python virtual environment
```bash
python3 -m venv ci_env
(for linux  ) source ci_env/bin/activate
(for windows) .\ci_env\Scripts\activate
```

Install with PIP
```
# install torch with GPU support for example:
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

# install clip-interrogator
pip install clip-interrogator==0.5.4

# or for very latest WIP with BLIP2 support
#pip install clip-interrogator==0.6.0
```

You can then use it in your script
```python
from PIL import Image
from clip_interrogator import Config, Interrogator
image = Image.open(image_path).convert('RGB')
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
print(ci.interrogate(image))
```

CLIP Interrogator uses OpenCLIP which supports many different pretrained CLIP models. For the best prompts for 
Stable Diffusion 1.X use `ViT-L-14/openai` for clip_model_name. For Stable Diffusion 2.0 use `ViT-H-14/laion2b_s32b_b79k`

## Configuration

The `Config` object lets you configure CLIP Interrogator's processing. 
* `clip_model_name`: which of the OpenCLIP pretrained CLIP models to use
* `cache_path`: path where to save precomputed text embeddings
* `download_cache`: when True will download the precomputed embeddings from huggingface
* `chunk_size`: batch size for CLIP, use smaller for lower VRAM
* `quiet`: when True no progress bars or text output will be displayed

On systems with low VRAM you can call `config.apply_low_vram_defaults()` to reduce the amount of VRAM needed (at the cost of some speed and quality). The default settings use about 6.3GB of VRAM and the low VRAM settings use about 2.7GB.

See the [run_cli.py](https://github.com/pharmapsychotic/clip-interrogator/blob/main/run_cli.py) and [run_gradio.py](https://github.com/pharmapsychotic/clip-interrogator/blob/main/run_gradio.py) for more examples on using Config and Interrogator classes.


## Ranking against your own list of terms (requires version 0.6.0)

```python
from clip_interrogator import Config, Interrogator, LabelTable, load_list
from PIL import Image

ci = Interrogator(Config(blip_model_type=None))
image = Image.open(image_path).convert('RGB')
table = LabelTable(load_list('terms.txt'), 'terms', ci)
best_match = table.rank(ci.image_to_features(image), top_count=1)[0]
print(best_match)
```
