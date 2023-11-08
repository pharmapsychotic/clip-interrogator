from typing import List

from tungstenkit import BaseIO, Field, Image, Option, define_model

from clip_interrogator import Config, Interrogator

CLIP_MODEL_NAMES = [
    "ViT-L-14/openai",
    "ViT-H-14/laion2b_s32b_b79k",
    "ViT-bigG-14/laion2b_s39b_b160k",
]


class Input(BaseIO):
    input_image: Image = Field(description="Input image")
    clip_model_name: str = Option(
        default="ViT-L-14/openai",
        choices=[
            "ViT-L-14/openai",
            "ViT-H-14/laion2b_s32b_b79k",
            "ViT-bigG-14/laion2b_s39b_b160k",
        ],
        description="Choose ViT-L for Stable Diffusion 1, ViT-H for Stable Diffusion 2, or ViT-bigG for Stable Diffusion XL.",
    )
    mode: str = Option(
        default="best",
        choices=["best", "classic", "fast", "negative"],
        description="Prompt mode (best takes 10-20 seconds, fast takes 1-2 seconds).",
    )


class Output(BaseIO):
    interrogated: str


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    cuda_version="11.8",
    python_version="3.10",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0"],
    python_packages=[
        "safetensors==0.3.3",
        "tqdm==4.66.1",
        "open_clip_torch==2.20.0",
        "accelerate==0.22.0",
        "transformers==4.33.1",
    ],
    batch_size=1,
)
class CLIPInterrogator:
    @staticmethod
    def post_build():
        """Download weights"""
        ci = Interrogator(
            Config(
                clip_model_name="ViT-L-14/openai",
                clip_model_path="cache",
                device="cpu",
            )
        )
        for clip_model_name in CLIP_MODEL_NAMES:
            ci.config.clip_model_name = clip_model_name
            ci.load_clip_model()

    def setup(self):
        """Load weights"""
        self.ci = Interrogator(
            Config(
                clip_model_name="ViT-L-14/openai",
                clip_model_path="cache",
                device="cuda:0",
            )
        )

    def predict(self, inputs: List[Input]) -> str:
        """Run a single prediction on the model"""
        input = inputs[0]
        image = input.input_image
        clip_model_name = input.clip_model_name
        mode = input.mode

        image = image.to_pil_image()
        self.switch_model(clip_model_name)
        if mode == "best":
            ret = self.ci.interrogate(image)
        elif mode == "classic":
            ret = self.ci.interrogate_classic(image)
        elif mode == "fast":
            ret = self.ci.interrogate_fast(image)
        elif mode == "negative":
            ret = self.ci.interrogate_negative(image)
        else:
            raise RuntimeError(f"Unknown mode: {ret}")

        return [Output(interrogated=ret)]

    def switch_model(self, clip_model_name: str):
        if clip_model_name != self.ci.config.clip_model_name:
            self.ci.config.clip_model_name = clip_model_name
            self.ci.load_clip_model()
