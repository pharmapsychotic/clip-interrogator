import sys
from PIL import Image
from cog import BasePredictor, Input, Path

from clip_interrogator import Config, Interrogator


class Predictor(BasePredictor):
    def setup(self):
        self.ci = Interrogator(Config(
            clip_model_name="ViT-L-14/openai",
            clip_model_path='cache',
            device='cuda:0', 
        ))

    def predict(
        self, 
        image: Path = Input(description="Input image"),
        clip_model_name: str = Input(
            default="ViT-L-14/openai",
            choices=["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k", "ViT-bigG-14/laion2b_s39b_b160k"],
            description="Choose ViT-L for Stable Diffusion 1, ViT-H for Stable Diffusion 2, or ViT-bigG for Stable Diffusion XL.",
        ),        
        mode: str = Input(
            default="best",
            choices=["best", "classic", "fast", "negative"],
            description="Prompt mode (best takes 10-20 seconds, fast takes 1-2 seconds).",
        ),        
    ) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(image)).convert("RGB")
        self.switch_model(clip_model_name)
        if mode == 'best':
            return self.ci.interrogate(image)
        elif mode == 'classic':
            return self.ci.interrogate_classic(image)
        elif mode == 'fast':
            return self.ci.interrogate_fast(image)
        elif mode == 'negative':
            return self.ci.interrogate_negative(image)

    def switch_model(self, clip_model_name: str):
        if clip_model_name != self.ci.config.clip_model_name:
            self.ci.config.clip_model_name = clip_model_name
            self.ci.load_clip_model()
