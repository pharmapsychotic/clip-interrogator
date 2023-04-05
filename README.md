# clip-interrogator

*Want to figure out what a good prompt might be to create new images like an existing one? The **CLIP Interrogator** is here to get you answers!*

## Run it!

🆕 Now available as a [Stable Diffusion Web UI Extension](https://github.com/pharmapsychotic/clip-interrogator-ext)! 🆕

<br>

Run Version 2 on Colab, HuggingFace, and Replicate!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb) [![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/pharma/CLIP-Interrogator) [![Replicate](https://replicate.com/pharmapsychotic/clip-interrogator/badge)](https://replicate.com/pharmapsychotic/clip-interrogator) [![Lambda](https://img.shields.io/badge/%CE%BB-Lambda-blue)](https://cloud.lambdalabs.com/demos/ml/CLIP-Interrogator)

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

## Deploying as Cloud Service (using Baseten)
This repo contains a [`truss`]("./truss"), which packages the model for cloud deployment using the [truss open-source library](https://github.com/basetenlabs/truss) by Baseten. Using this truss, you can easily deploy your own scalable cloud service of this model by following these steps.

1. Clone the repo: `git clone https://github.com/pharmapsychotic/clip-interrogator.git`
2. `cd clip-interrogator`
3. Setup virtualenv with baseten and truss deps (make sure to upgrade)
```
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install --upgrade baseten truss
```
4. [Grab API key from your Baseten account](https://docs.baseten.co/settings/api-keys)
5. Deploy using this command
```
BASETEN_API_KEY=API_KEY_COPIED_FROM_BASETEN python deploy_baseten.py
```
6. You'll get an email once your model is ready and you can call it using the instructions from the UI.
Below is a sample invocation.
```
import baseten, os
baseten.login(os.environ["BASETEN_API_KEY"])

img_str = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII='

model = baseten.deployed_model_id("MODEL_ID_FROM_ACCOUNT")
model.predict({
    "image": img_str,
    "format": "PNG",
    "mode": "fast",
    "clip_model_name": "ViT-L-14/openai"
})
```