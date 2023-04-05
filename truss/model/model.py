from typing import Dict, List

import torch
from b64_utils import b64_to_pil
from clip_interrogator import Config, Interrogator

DEFAULT_MODEL_NAME = "ViT-L-14/openai"


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.ci = None

    def load(self):
        self.ci = Interrogator(
            Config(
                clip_model_name=DEFAULT_MODEL_NAME,
                clip_model_path="cache",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

    def switch_model(self, clip_model_name: str):
        if clip_model_name != self.ci.config.clip_model_name:
            self.ci.config.clip_model_name = clip_model_name
            self.ci.load_clip_model()

    def inference(self, image, mode) -> str:
        image = image.convert("RGB")
        if mode == "best":
            return self.ci.interrogate(image)
        elif mode == "classic":
            return self.ci.interrogate_classic(image)
        elif mode == "fast":
            return self.ci.interrogate_fast(image)
        elif mode == "negative":
            return self.ci.interrogate_negative(image)
        raise ValueError(f"unsupported mode: {mode}")

    def predict(self, request: Dict) -> Dict[str, List]:
        image_b64 = request.pop("image")
        image_fmt = request.get("format", "PNG")
        image = b64_to_pil(image_b64, format=image_fmt)
        mode = request.get("mode", "fast")
        clip_model_name = request.get("clip_model_name", DEFAULT_MODEL_NAME)
        self.switch_model(clip_model_name)

        return {"caption": self.inference(image, mode)}
