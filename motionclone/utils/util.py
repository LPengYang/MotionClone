import hashlib
import io
import re
import os
import imageio
import numpy as np
from typing import Union

import cv2
import numpy as np
import requests
import random
import torch
import PIL.Image
import PIL.ImageOps
from PIL import Image
from typing import Callable, Union

import torch
import torchvision
import torch.distributed as dist
import torch.nn.functional as F
import decord
decord.bridge.set_bridge('torch')
from PIL import Image, ImageOps

from safetensors import safe_open
# from tqdm import tqdm
from einops import rearrange
from motionclone.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint,convert_ldm_clip_checkpoint_concise
from motionclone.utils.convert_lora_safetensor_to_diffusers import convert_lora, load_diffusers_lora
from huggingface_hub import snapshot_download
# from transformers import (
#     AutoFeatureExtractor,
#     BertTokenizerFast,
#     CLIPImageProcessor,
#     CLIPTextConfig,
#     CLIPTextModel,
#     CLIPTextModelWithProjection,
#     CLIPTokenizer,
#     CLIPVisionConfig,
#     CLIPVisionModelWithProjection,
# )

MOTION_MODULES = [
    "mm_sd_v14.ckpt", 
    "mm_sd_v15.ckpt", 
    "mm_sd_v15_v2.ckpt", 
    "v3_sd15_mm.ckpt",
]

ADAPTERS = [
    # "mm_sd_v14.ckpt",
    # "mm_sd_v15.ckpt",
    # "mm_sd_v15_v2.ckpt",
    # "mm_sdxl_v10_beta.ckpt",
    "v2_lora_PanLeft.ckpt",
    "v2_lora_PanRight.ckpt",
    "v2_lora_RollingAnticlockwise.ckpt",
    "v2_lora_RollingClockwise.ckpt",
    "v2_lora_TiltDown.ckpt",
    "v2_lora_TiltUp.ckpt",
    "v2_lora_ZoomIn.ckpt",
    "v2_lora_ZoomOut.ckpt",
    "v3_sd15_adapter.ckpt",
    # "v3_sd15_mm.ckpt",
    "v3_sd15_sparsectrl_rgb.ckpt",
    "v3_sd15_sparsectrl_scribble.ckpt",
]

BACKUP_DREAMBOOTH_MODELS = [
    "realisticVisionV60B1_v51VAE.safetensors",
    "majicmixRealistic_v4.safetensors",
    "leosamsFilmgirlUltra_velvia20Lora.safetensors",
    "toonyou_beta3.safetensors",
    "majicmixRealistic_v5Preview.safetensors",
    "rcnzCartoon3d_v10.safetensors",
    "lyriel_v16.safetensors",
    "leosamsHelloworldXL_filmGrain20.safetensors",
    "TUSUN.safetensors",
]

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def auto_download(local_path, is_dreambooth_lora=False):
    hf_repo = "guoyww/animatediff_t2i_backups" if is_dreambooth_lora else "guoyww/animatediff"
    folder, filename = os.path.split(local_path)

    if not os.path.exists(local_path):
        print(f"local file {local_path} does not exist. trying to download from {hf_repo}")

        if is_dreambooth_lora: assert filename in BACKUP_DREAMBOOTH_MODELS, f"{filename} dose not exist in {hf_repo}"
        else: assert filename in MOTION_MODULES + ADAPTERS, f"{filename} dose not exist in {hf_repo}"

        folder = "." if folder == "" else folder
        os.makedirs(folder, exist_ok=True)
        snapshot_download(repo_id=hf_repo, local_dir=folder, allow_patterns=[filename])

def load_weights(
    animation_pipeline,
    # motion module
    motion_module_path         = "",
    motion_module_lora_configs = [],
    # domain adapter
    adapter_lora_path          = "",
    adapter_lora_scale         = 1.0,
    # image layers
    dreambooth_model_path      = "",
    lora_model_path            = "",
    lora_alpha                 = 0.8,
):
    # motion module
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
        unet_state_dict.pop("animatediff_config", "")
    
    missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    # assert len(unexpected) == 0
    del unet_state_dict

    # base model
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            # import pdb; pdb.set_trace()
            dreambooth_state_dict = {}
            # import safetensors
            # dreambooth_state_dict = safetensors.torch.load_file(dreambooth_model_path)
            # import pdb; pdb.set_trace()
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
            # import pdb; pdb.set_trace()
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
            
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
        animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
        animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        
        # 3. text_model
        # animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_concise(dreambooth_state_dict)
        animation_pipeline.text_encoder.load_state_dict(converted_text_encoder_checkpoint, strict=True)
        del dreambooth_state_dict, converted_vae_checkpoint, converted_unet_checkpoint, converted_text_encoder_checkpoint
        
        # clip_config_name = "models/clip-vit-large-patch14"
        # clip_config = CLIPTextConfig.from_pretrained(clip_config_name, local_files_only=True)
        # text_model = CLIPTextModel(clip_config)
        # keys = list(dreambooth_state_dict.keys())
        # text_model_dict = {}
        # for key in keys:
        #     if key.startswith("cond_stage_model.transformer"):
        #         text_model_dict[key[len("cond_stage_model.transformer.") :]] = dreambooth_state_dict[key]  
        # text_model.load_state_dict(text_model_dict)
        # animation_pipeline.text_encoder = text_model.to(dtype=animation_pipeline.unet.dtype)
        # # import pdb; pdb.set_trace()
        # # animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        # del dreambooth_state_dict
        
    # lora layers
    if lora_model_path != "":
        print(f"load lora model from {lora_model_path}")
        assert lora_model_path.endswith(".safetensors")
        lora_state_dict = {}
        with safe_open(lora_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)
                
        animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
        del lora_state_dict

    # domain adapter lora
    if adapter_lora_path != "":
        print(f"load domain lora from {adapter_lora_path}")
        domain_lora_state_dict = torch.load(adapter_lora_path, map_location="cpu")
        domain_lora_state_dict = domain_lora_state_dict["state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
        domain_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, domain_lora_state_dict, alpha=adapter_lora_scale)

    # motion module lora
    for motion_module_lora_config in motion_module_lora_configs:
        path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
        print(f"load motion LoRA from {path}")
        motion_lora_state_dict = torch.load(path, map_location="cpu")
        motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
        motion_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, motion_lora_state_dict, alpha)

    return animation_pipeline

def video_preprocess(video_path, height, width, video_length, duration=None, sample_start_idx=0,):
    
    video_name = video_path.split('/')[-1].split('.')[0]
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    if  duration is None:
        # 读取整个视频
        total_frames = len(vr)
    else:
        # 根据给定的时长（秒）计算帧数
        total_frames = int(fps * duration)
        total_frames = min(total_frames, len(vr))  # 确保不超过视频总长度
        
    sample_index = np.linspace(0, total_frames - 1, video_length, dtype=int)
    print(total_frames,sample_index)
    video = vr.get_batch(sample_index)
    video = rearrange(video, "f h w c -> f c h w")

    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=True)
    
    # video_sample = rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
    # imageio.mimwrite(f"processed_videos/sample_{video_name}.mp4", video_sample[0], fps=8, quality=9)
    
    video = video / 127.5 - 1.0

    return video


def set_nested_item(dataDict, mapList, value):
    """Set item in nested dictionary"""
    """
    Example: the mapList contains the name of each key ['injection','self-attn']
            this method will change the content in dataDict['injection']['self-attn'] with value

    """
    for k in mapList[:-1]:
        dataDict = dataDict[k]
    dataDict[mapList[-1]] = value


def merge_sweep_config(base_config, update):
    """Merge the updated parameters into the base config"""

    if base_config is None:
        raise ValueError("Base config is None")
    if update is None:
        raise ValueError("Update config is None")
    for key in update.keys():
        map_list = key.split("--")
        set_nested_item(base_config, map_list, update[key])
    return base_config


# Adapt from https://github.com/castorini/daam
def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    merge_idxs = []
    tokens = tokenizer.tokenize(prompt.lower())
    if word_idx is None:
        word = word.lower()
        search_tokens = tokenizer.tokenize(word)
        start_indices = [x + offset_idx for x in range(len(tokens)) if
                         tokens[x:x + len(search_tokens)] == search_tokens]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise Exception(f'Search word {word} not found in prompt!')
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.


def extract_data(input_string: str) -> list:
    print("input_string:", input_string)
    """
    Extract data from a string pattern where contents in () are separated by ;
    The first item in each () is considered as 'ref' and the rest as 'gen'.

    Args:
    - input_string (str): The input string pattern.

    Returns:
    - list: A list of dictionaries containing 'ref' and 'gen'.
    """
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, input_string)

    data = []
    for match in matches:
        parts = [x.strip() for x in match.split(';')]
        ref = parts[0].strip()
        gen = parts[1].strip()
        data.append({'ref': ref, 'gen': gen})

    return data


def generate_hash_key(image, prompt=""):
    """
    Generate a hash key for the given image and prompt.
    """
    byte_array = io.BytesIO()
    image.save(byte_array, format='JPEG')

    # Get byte data
    image_byte_data = byte_array.getvalue()

    # Combine image byte data and prompt byte data
    combined_data = image_byte_data + prompt.encode('utf-8')

    sha256 = hashlib.sha256()
    sha256.update(combined_data)
    return sha256.hexdigest()


def save_data(data, folder_path, key):
    """
    Save data to a file, using key as the file name
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{key}.pt")

    torch.save(data, file_path)


def get_data(folder_path, key):
    """
    Get data from a file, using key as the file name
    :param folder_path:
    :param key:
    :return:
    """

    file_path = os.path.join(folder_path, f"{key}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        return None


def PILtoTensor(data: Image.Image) -> torch.Tensor:
    return torch.tensor(np.array(data)).permute(2, 0, 1).unsqueeze(0).float()


def TensorToPIL(data: torch.Tensor) -> Image.Image:
    return Image.fromarray(data.squeeze().permute(1, 2, 0).numpy().astype(np.uint8))

# Adapt from https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/utils/loading_utils.py#L9
def load_image(
        image: Union[str, PIL.Image.Image], convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image

# Take from huggingface/diffusers
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def _in_step(config, step):
    in_step = False
    try:
        start_step = config.start_step
        end_step = config.end_step
        if start_step <= step < end_step:
            in_step = True
    except:
        in_step = False
    return in_step

def classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True