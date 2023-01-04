import sys
from PIL import Image
from cog import BasePredictor, Input, Path

from clip_interrogator import Interrogator, Config


class Predictor(BasePredictor):
    def setup(self):
        self.ci = Interrogator(Config(
            blip_model_url='cache/model_large_caption.pth',
            clip_model_name="ViT-L-14/openai",
            clip_model_path='cache',
            device='cuda:0', 
        ))

    def predict(
        self, 
        image: Path = Input(description="Input image"),
        clip_model_name: str = Input(
            default="ViT-L-14/openai",
            choices=["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"],
            description="Choose ViT-L for Stable Diffusion 1, and ViT-H for Stable Diffusion 2",
        ),        
        mode: str = Input(
            default="best",
            choices=["best", "fast"],
            description="Prompt mode (best takes 10-20 seconds, fast takes 1-2 seconds).",
        ),        
    ) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(image)).convert("RGB")
        self.switch_model(clip_model_name)
        if mode == "best":
            return self.ci.interrogate(image)
        else:
            return self.ci.interrogate_fast(image)
    
    def switch_model(self, clip_model_name: str):
        if clip_model_name != self.ci.config.clip_model_name:
            self.ci.config.clip_model_name = clip_model_name
            self.ci.load_clip_model()
