import os
import json
import torch
import argparse
import imageio
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange, repeat
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from motionclone.models.unet import UNet3DConditionModel
from motionclone.pipelines.pipeline import MotionClonePipeline
from motionclone.pipelines.additional_components import customized_step, set_timesteps
from motionclone.utils.util import load_weights
from motionclone.utils.util import set_all_seed

def main(args):

    if not os.path.exists("samples"):
        os.makedirs("samples")
    
    config  = OmegaConf.load(args.config)
    adopted_dtype = torch.float16
    device = "cuda"

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)

    # for model_idx, model_config in enumerate(config):
    config.width = config.get("W", args.W)
    config.height = config.get("H", args.H)
    config.video_length = config.get("L", args.L)

    model_config = OmegaConf.load(config.get("model_config", args.model_config))
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs)).to(device).to(dtype=adopted_dtype)
    
    controlnet = None
    # set xformers
    if is_xformers_available() and (not args.without_xformers):
        unet.enable_xformers_memory_efficient_attention()
        if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()
    pipeline = MotionClonePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet,
        scheduler = DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)),
    ).to(device)

    pipeline = load_weights(
        pipeline,
        # motion module
        motion_module_path         = config.get("motion_module", ""),
        dreambooth_model_path      = config.get("dreambooth_path", ""),
    ).to(device)
    
    pipeline.scheduler.customized_step = customized_step.__get__(pipeline.scheduler)
    pipeline.scheduler.added_set_timesteps = set_timesteps.__get__(pipeline.scheduler)
    # use low vram
    pipeline.enable_sequential_cpu_offload()
    
    seed = config.get("seed", args.default_seed)
    set_all_seed(seed)
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(seed)
    pipeline.scheduler.added_set_timesteps(config.num_inference_step, device=device)
    if args.examples == None:
        video_name = config.video_path.split('/')[-1].split('.')[0]
        inversion_data_path =  os.path.join(args.inversion_save_dir, f"inverted_data_{video_name}.pkl")
        videos = pipeline(
                        config = config,
                        generator = generator,
                        inversion_data_path = inversion_data_path
                    )
        videos = rearrange(videos, "b c f h w -> b f h w c")
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = os.path.join(args.save_dir,  config.new_prompt.replace(' ', '_') + f"_seed{seed}_up12_motion" + '.mp4')
        videos_uint8 = (videos[0] * 255).astype(np.uint8)
        imageio.mimwrite(save_path, videos_uint8, fps=8, quality=9)
        print(config.guidance_step, config.num_inference_step,)
        print(config.temp_guidance)
        print(config.app_guidance)
        print(save_path)
    else:
        examples = json.load(args.examples)
        for example_infor in examples:
            config.video_path = example_infor["video_path"]
            config.inversion_prompt = example_infor["inversion_prompt"]
            config.new_prompt = example_infor["new_prompt"]
            inversion_data_path =  os.path.join(args.inversion_save_dir, config.new_prompt.replace(' ', '_') + ".pkl")
            videos = pipeline(
                        config = config,
                        generator = generator,
                        inversion_data_path = inversion_data_path,
                    )
            videos = rearrange(videos, "b c f h w -> b f h w c")
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir,  config.new_prompt.replace(' ', '_') + f"_seed{seed}_up12_motion" + '.mp4')
            videos_uint8 = (videos[0] * 255).astype(np.uint8)
            imageio.mimwrite(save_path, videos_uint8, fps=8, quality=9)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--model-config",      type=str, default="configs/model_config.yaml")    
    parser.add_argument("--config",            type=str, default="configs/example.yaml")
    parser.add_argument("--examples",          type=str, default=None)
    parser.add_argument("--save_dir",          type=str, default="samples/")
    parser.add_argument("--inversion_save_dir",type=str, default="inversion/")
    
    parser.add_argument("--default-seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
