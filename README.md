# clip-interrogator-with-less-VRAM

*Want to figure out what a good prompt might be to create new images like an existing one? The **CLIP Interrogator** is here to get you answers!*

This version uses less VRAM than the main repo by only having one model loaded at a time.

When you create an `Interrogator`:
```py
ci = Interrogator(Config())
```

The BLIP and CLIP models are both loaded, but only BLIP is on the GPU, the CLIP stays in RAM.

When you actually do inference:
```py
ci.interrogate(image)
# Or:
# ci.interrogate_classic(image)
# ci.interrogate_fast(image)
```

BLIP inference is done, it gets unloaded then CLIP gets loaded and infers.
If you run it again, CLIP is done first, then BLIP is loaded, to reduce pointless loading and unloading.

By using this, it (`classic` or `fast`, normal doesn't quite fit) can be run on as little as 4GB of VRAM, the main repo needing at least 6GB.

> But wouldn't loading a new model every time I want to interrogate an image be terrible for performance?

\- me

Absolutely.

There's little performance overhead for just one interrogation, since it's essentially lazy loading the CLIP model, but for multiple images, there will be a noticable effect.

That's why I made the `interrogate_batch` functions:
```py
# files = Some list of strings
images = [Image.open(f).convert("RGB") for f in files]
ci.interrogate_batch(images)
```

This does BLIP inference on each of the images, *then* loads the CLIP model, saving some performance.
There are also `interrogate_{classic,fast}_batch` functions.

## Run it!

Bash (linux/unix):
```sh
$ ./run_cli.py -i input.png -m $MODE
```

Windows:
```cmd
$ python run_cli.py -i input.png -m $MODE
```

Where `$MODE` is either `best`, `classic` or `fast` (default `best`)

Run Version 2 on Colab, HuggingFace, and Replicate!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb) [![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/pharma/CLIP-Interrogator) [![Replicate](https://replicate.com/pharmapsychotic/clip-interrogator/badge)](https://replicate.com/pharmapsychotic/clip-interrogator)

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
pip install clip-interrogator==0.5.1
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

See the [run_cli.py](https://github.com/pharmapsychotic/clip-interrogator/blob/main/run_cli.py) and [run_gradio.py](https://github.com/pharmapsychotic/clip-interrogator/blob/main/run_gradio.py) for more examples on using Config and Interrogator classes.
