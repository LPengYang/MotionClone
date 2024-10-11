import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from motionclone.models.unet import UNet3DConditionModel
from motionclone.pipelines.pipeline_animation import AnimationPipeline
from motionclone.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available
from motionclone.utils.motionclone_functions import *
import json
from motionclone.utils.xformer_attention import *

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu or str(os.getenv('CUDA_VISIBLE_DEVICES', 0))
    
    config  = OmegaConf.load(args.inference_config)
    adopted_dtype = torch.float16
    device = "cuda"
    set_all_seed(42)

    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)
    
    config.width = config.get("W", args.W)
    config.height = config.get("H", args.H)
    config.video_length = config.get("L", args.L)
    
    if not os.path.exists(args.generated_videos_save_dir):
        os.makedirs(args.generated_videos_save_dir)
    OmegaConf.save(config, os.path.join(args.generated_videos_save_dir,"inference_config.json"))
    
    model_config = OmegaConf.load(config.get("model_config", ""))
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs),).to(device).to(dtype=adopted_dtype)
    
    # set xformers
    if is_xformers_available() and (not args.without_xformers):
        unet.enable_xformers_memory_efficient_attention()

    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=None,
        scheduler=DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)),
    ).to(device)
    
    pipeline = load_weights(
        pipeline,
        # motion module
        motion_module_path         = config.get("motion_module", ""),
        dreambooth_model_path      = config.get("dreambooth_path", ""),
    ).to(device)
    pipeline.text_encoder.to(dtype=adopted_dtype)
    
    # load customized functions from motionclone_functions
    pipeline.scheduler.customized_step = schedule_customized_step.__get__(pipeline.scheduler)
    pipeline.scheduler.customized_set_timesteps = schedule_set_timesteps.__get__(pipeline.scheduler)
    pipeline.unet.forward = unet_customized_forward.__get__(pipeline.unet)
    pipeline.sample_video = sample_video.__get__(pipeline)
    pipeline.single_step_video = single_step_video.__get__(pipeline)
    pipeline.get_temp_attn_prob = get_temp_attn_prob.__get__(pipeline)
    pipeline.add_noise = add_noise.__get__(pipeline)
    pipeline.compute_temp_loss = compute_temp_loss.__get__(pipeline)
    pipeline.obtain_motion_representation = obtain_motion_representation.__get__(pipeline)
    
    for param in pipeline.unet.parameters():
        param.requires_grad = False
    pipeline.input_config,  pipeline.unet.input_config = config,  config
    
    pipeline.unet = prep_unet_attention(pipeline.unet,pipeline.input_config.motion_guidance_blocks)
    pipeline.unet = prep_unet_conv(pipeline.unet)
    pipeline.scheduler.customized_set_timesteps(config.inference_steps, config.guidance_steps,config.guidance_scale,device=device,timestep_spacing_type = "uneven")
    # pipeline.scheduler.customized_set_timesteps(config.inference_steps,device=device,timestep_spacing_type = "linspace")
    with open(args.examples, 'r') as files:
        for line in files:
            # prepare infor of each case
            example_infor = json.loads(line)
            config.video_path = example_infor["video_path"]
            config.new_prompt = example_infor["new_prompt"] + config.get("positive_prompt", "")
            pipeline.input_config,  pipeline.unet.input_config = config,  config  # update config
            
            #  perform motion representation extraction
            seed_motion = example_infor.get("seed", args.default_seed) 
            generator = torch.Generator(device=pipeline.device)
            generator.manual_seed(seed_motion)
            if not os.path.exists(args.motion_representation_save_dir):
                os.makedirs(args.motion_representation_save_dir)
            motion_representation_path = os.path.join(args.motion_representation_save_dir, os.path.splitext(os.path.basename(config.video_path))[0] + '.pt') 
            pipeline.obtain_motion_representation(generator= generator, motion_representation_path = motion_representation_path) 
            
            # perform video generation 
            seed = seed_motion # can assign other seed here
            generator = torch.Generator(device=pipeline.device)
            generator.manual_seed(seed)
            pipeline.input_config.seed = seed
            
            videos = pipeline.sample_video(generator = generator,)
            videos = rearrange(videos, "b c f h w -> b f h w c")
            save_path = os.path.join(args.generated_videos_save_dir,  os.path.splitext(os.path.basename(config.video_path))[0] 
                                    + "_" + config.new_prompt.strip().replace(' ', '_') + str(seed_motion) + "_" +str(seed)+'.mp4')                                        
            videos_uint8 = (videos[0] * 255).astype(np.uint8)

            imageio.mimwrite(save_path, videos_uint8, fps=8)
            print(save_path,"is done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion",)
        
    parser.add_argument("--inference_config",                type=str, default="configs/t2v_camera.yaml")
    parser.add_argument("--examples",                type=str, default="configs/t2v_camera.jsonl")
    parser.add_argument("--motion-representation-save-dir",      type=str, default="motion_representation/")
    parser.add_argument("--generated-videos-save-dir",                type=str, default="generated_videos")
    
    parser.add_argument("--visible_gpu", type=str, default=None)
    parser.add_argument("--default-seed", type=int, default=2025)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
