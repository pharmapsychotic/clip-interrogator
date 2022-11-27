import sys
from PIL import Image
from cog import BasePredictor, Input, Path

sys.path.extend(["src/clip", "src/blip"])

from clip_interrogator import Interrogator, Config


class Predictor(BasePredictor):
    def setup(self):
        config = Config(device="cuda:0", clip_model_name='ViT-L-14/openai')
        self.ci = Interrogator(config)

    def predict(self, image: Path = Input(description="Input image")) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(image)).convert("RGB")
        return self.ci.interrogate(image)
