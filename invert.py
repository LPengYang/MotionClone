import os
import json
import argparse
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms


from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from motionclone.models.unet import UNet3DConditionModel
from motionclone.pipelines.pipeline import MotionClonePipeline
from motionclone.pipelines.additional_components import customized_step, set_timesteps
from motionclone.utils.util import load_weights
from motionclone.utils.util import video_preprocess


@torch.no_grad()
def main(args):

    if not os.path.exists("inversion"):
        os.makedirs("inversion")
    
    config  = OmegaConf.load(args.config)
    adopted_dtype = torch.float16
    device = "cuda"
    
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)
    
    config.W = config.get("W", args.W)
    config.H = config.get("H", args.H)
    config.L = config.get("L", args.L)

    model_config = OmegaConf.load(config.get("model_config", args.model_config))
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs),).to(device).to(dtype=adopted_dtype)
    
    controlnet = None
    # set xformers
    if is_xformers_available() and (not args.without_xformers):
        unet.enable_xformers_memory_efficient_attention()
        if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

    pipeline = MotionClonePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)),
    ).to(device)
    
    pipeline.scheduler.added_set_timesteps = set_timesteps.__get__(pipeline.scheduler)
    config.num_inference_step = 1000
    pipeline.scheduler.added_set_timesteps(config.num_inference_step)
    
    pipeline = load_weights(
        pipeline,
        # motion module
        motion_module_path         = config.get("motion_module", ""),
        dreambooth_model_path      = config.get("dreambooth_path", ""),
    ).to(device)
    
    if args.examples == None:
        cond_video = video_preprocess(config)
        video_name = config.video_path.split('/')[-1].split('.')[0]
        save_inversion_path =  os.path.join(args.inversion_save_dir, f"inverted_data_{video_name}.pkl")
        pipeline.invert(video = cond_video, config = config, save_path = save_inversion_path)
    else:
        examples = json.load(args.examples)
        for example_infor in examples:
            config.video_path = example_infor["video_path"]
            config.inversion_prompt = example_infor["inversion_prompt"]
            config.new_prompt = example_infor["new_prompt"]
            cond_video = video_preprocess(config)
            pipeline.invert(video = cond_video, config = config)
            
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path",   type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--model-config",            type=str, default="configs/model_config/model_config.yaml")    
    parser.add_argument("--config",                  type=str, default="configs/example.yaml")
    parser.add_argument("--inversion_save_dir",      type=str, default="inversion/")
    parser.add_argument("--examples",                type=str, default=None)
    
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
