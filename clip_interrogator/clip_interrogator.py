import hashlib
import inspect
import math
import numpy as np
import open_clip
import os
import pickle
import time
import torch

from dataclasses import dataclass
from models.blip import blip_decoder
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from typing import List


@dataclass 
class Config:
    # models can optionally be passed in directly
    blip_model = None
    clip_model = None
    clip_preprocess = None

    # blip settings
    blip_image_eval_size: int = 384
    blip_max_length: int = 32
    blip_model_url: str = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    blip_num_beams: int = 8
    blip_offload: bool = False

    # clip settings
    clip_model_name: str = 'ViT-L-14/openai'
    clip_model_path: str = None

    # interrogator settings
    cache_path: str = 'cache'
    chunk_size: int = 2048
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    flavor_intermediate_count: int = 2048
    quiet: bool = False # when quiet progress bars are not shown


class Interrogator():
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        if config.blip_model is None:
            if not config.quiet:
                print("Loading BLIP model...")
            blip_path = os.path.dirname(inspect.getfile(blip_decoder))
            configs_path = os.path.join(os.path.dirname(blip_path), 'configs')
            med_config = os.path.join(configs_path, 'med_config.json')
            blip_model = blip_decoder(
                pretrained=config.blip_model_url, 
                image_size=config.blip_image_eval_size, 
                vit='large', 
                med_config=med_config
            )
            blip_model.eval()
            blip_model = blip_model.to(config.device)
            self.blip_model = blip_model
        else:
            self.blip_model = config.blip_model

        self.load_clip_model()

    def load_clip_model(self):
        start_time = time.time()
        config = self.config

        if config.clip_model is None:
            if not config.quiet:
                print("Loading CLIP model...")

            clip_model_name, clip_model_pretrained_name = config.clip_model_name.split('/', 2)
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name, 
                pretrained=clip_model_pretrained_name, 
                precision='fp16' if config.device == 'cuda' else 'fp32',
                device=config.device,
                jit=False,
                cache_dir=config.clip_model_path
            )
            self.clip_model.to(config.device).eval()
        else:
            self.clip_model = config.clip_model
            self.clip_preprocess = config.clip_preprocess
        self.tokenize = open_clip.get_tokenizer(clip_model_name)

        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
        trending_list = [site for site in sites]
        trending_list.extend(["trending on "+site for site in sites])
        trending_list.extend(["featured on "+site for site in sites])
        trending_list.extend([site+" contest winner" for site in sites])

        raw_artists = _load_list(config.data_path, 'artists.txt')
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])

        self.artists = LabelTable(artists, "artists", self.clip_model, self.tokenize, config)
        self.flavors = LabelTable(_load_list(config.data_path, 'flavors.txt'), "flavors", self.clip_model, self.tokenize, config)
        self.mediums = LabelTable(_load_list(config.data_path, 'mediums.txt'), "mediums", self.clip_model, self.tokenize, config)
        self.movements = LabelTable(_load_list(config.data_path, 'movements.txt'), "movements", self.clip_model, self.tokenize, config)
        self.trendings = LabelTable(trending_list, "trendings", self.clip_model, self.tokenize, config)

        end_time = time.time()
        if not config.quiet:
            print(f"Loaded CLIP model and data in {end_time-start_time:.2f} seconds.")

    def generate_caption(self, pil_image: Image) -> str:
        if self.config.blip_offload:
            self.blip_model = self.blip_model.to(self.device)
        size = self.config.blip_image_eval_size
        gpu_image = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            caption = self.blip_model.generate(
                gpu_image, 
                sample=False, 
                num_beams=self.config.blip_num_beams, 
                max_length=self.config.blip_max_length, 
                min_length=5
            )
        if self.config.blip_offload:
            self.blip_model = self.blip_model.to("cpu")
        return caption[0]

    def image_to_features(self, image: Image) -> torch.Tensor:
        images = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def interrogate_classic(self, image: Image, max_flavors: int=3) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)

        medium = self.mediums.rank(image_features, 1)[0]
        artist = self.artists.rank(image_features, 1)[0]
        trending = self.trendings.rank(image_features, 1)[0]
        movement = self.movements.rank(image_features, 1)[0]
        flaves = ", ".join(self.flavors.rank(image_features, max_flavors))

        if caption.startswith(medium):
            prompt = f"{caption} {artist}, {trending}, {movement}, {flaves}"
        else:
            prompt = f"{caption}, {medium} {artist}, {trending}, {movement}, {flaves}"

        return _truncate_to_fit(prompt, self.tokenize)

    def interrogate_fast(self, image: Image, max_flavors: int = 32) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)
        merged = _merge_tables([self.artists, self.flavors, self.mediums, self.movements, self.trendings], self.config)
        tops = merged.rank(image_features, max_flavors)
        return _truncate_to_fit(caption + ", " + ", ".join(tops), self.tokenize)

    def interrogate(self, image: Image, max_flavors: int=32) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)

        flaves = self.flavors.rank(image_features, self.config.flavor_intermediate_count)
        best_medium = self.mediums.rank(image_features, 1)[0]
        best_artist = self.artists.rank(image_features, 1)[0]
        best_trending = self.trendings.rank(image_features, 1)[0]
        best_movement = self.movements.rank(image_features, 1)[0]

        best_prompt = caption
        best_sim = self.similarity(image_features, best_prompt)

        def check(addition: str) -> bool:
            nonlocal best_prompt, best_sim
            prompt = best_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if sim > best_sim:
                best_sim = sim
                best_prompt = prompt
                return True
            return False

        def check_multi_batch(opts: List[str]):
            nonlocal best_prompt, best_sim
            prompts = []
            for i in range(2**len(opts)):
                prompt = best_prompt
                for bit in range(len(opts)):
                    if i & (1 << bit):
                        prompt += ", " + opts[bit]
                prompts.append(prompt)

            t = LabelTable(prompts, None, self.clip_model, self.tokenize, self.config)
            best_prompt = t.rank(image_features, 1)[0]
            best_sim = self.similarity(image_features, best_prompt)

        check_multi_batch([best_medium, best_artist, best_trending, best_movement])

        extended_flavors = set(flaves)
        for _ in tqdm(range(max_flavors), desc="Flavor chain", disable=self.config.quiet):
            best = self.rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
            flave = best[len(best_prompt)+2:]
            if not check(flave):
                break
            if _prompt_at_max_len(best_prompt, self.tokenize):
                break
            extended_flavors.remove(flave)

        return best_prompt

    def rank_top(self, image_features: torch.Tensor, text_array: List[str]) -> str:
        text_tokens = self.tokenize([text for text in text_array]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return text_array[similarity.argmax().item()]

    def similarity(self, image_features: torch.Tensor, text: str) -> float:
        text_tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity[0][0].item()


class LabelTable():
    def __init__(self, labels:List[str], desc:str, clip_model, tokenize, config: Config):
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeds = []
        self.labels = labels
        self.tokenize = tokenize

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()

        cache_filepath = None
        if config.cache_path is not None and desc is not None:
            os.makedirs(config.cache_path, exist_ok=True)
            sanitized_name = config.clip_model_name.replace('/', '_').replace('@', '_')
            cache_filepath = os.path.join(config.cache_path, f"{sanitized_name}_{desc}.pkl")
            if desc is not None and os.path.exists(cache_filepath):
                with open(cache_filepath, 'rb') as f:
                    try:
                        data = pickle.load(f)
                        if data.get('hash') == hash:
                            self.labels = data['labels']
                            self.embeds = data['embeds']
                            if self.device == 'cpu':
                                self.embeds = [e.astype(np.float32) for e in self.embeds]
                    except Exception as e:
                        print(f"Error loading cached table {desc}: {e}")

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels)/config.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.config.quiet):
                text_tokens = self.tokenize(chunk).to(self.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text_features = clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if cache_filepath is not None:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump({
                        "labels": self.labels, 
                        "embeds": self.embeds, 
                        "hash": hash, 
                        "model": config.clip_model_name
                    }, f)
    
    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int=1) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        with torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int=1) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.config.quiet):
            start = chunk_idx*self.chunk_size
            stop = min(start+self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start+i] for i in tops])
            top_embeds.extend([self.embeds[start+i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]


def _load_list(data_path: str, filename: str) -> List[str]:
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def _merge_tables(tables: List[LabelTable], config: Config) -> LabelTable:
    m = LabelTable([], None, None, None, config)
    for table in tables:
        m.labels.extend(table.labels)
        m.embeds.extend(table.embeds)
    return m

def _prompt_at_max_len(text: str, tokenize) -> bool:
    tokens = tokenize([text])
    return tokens[0][-1] != 0

def _truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        if _prompt_at_max_len(new_text + part, tokenize):
            break
        new_text += ', ' + part
    return new_text
