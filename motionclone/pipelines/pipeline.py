# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union, Any, Dict
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import einops
import imageio
import matplotlib.pyplot as plt
import yaml


from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb

from ..utils.xformer_attention import *
from ..utils.conv_layer import *
from ..utils.util import *
from ..utils.util import _in_step, _classify_blocks, ddim_inversion

from .additional_components import *

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class MotionClonePipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class MotionClonePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[SparseControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)
        

        
    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def invert(self,
               video = None,
               config: omegaconf.dictconfig = None,
               save_path = None,
               ):
        # perform DDIM inversion 
        import time
        start_time = time.time()
        generator = None
        video_latent = self.vae.encode(video.to(self.vae.dtype).to(self.vae.device)).latent_dist.sample(generator)
        video_latent = self.vae.config.scaling_factor * video_latent
        video_latent = video_latent.unsqueeze(0)
        video_latent = einops.rearrange(video_latent, "b f c h w -> b c f h w")                                                                 
        ddim_latents_dict, cond_embeddings = ddim_inversion(self, self.scheduler, video_latent, config.num_inference_step, config.inversion_prompt)
        
        end_time = time.time()
        # import pdb; pdb.set_trace()
        print("Inversion time", end_time - start_time)

        video_data: Dict = {
            'inversion_prompt': config.inversion_prompt,
            'all_latents_inversion': ddim_latents_dict,
            'raw_video': video,
            'inversion_prompt_embeds': cond_embeddings,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(video_data, f)
                
    
    def single_step(self,latents, i, t,all_latents,text_embeddings,ref_prompt_embeds,latents_dtype,cond_control_ids,cond_example_ids,cond_appearance_ids,
                    num_control_samples,do_classifier_free_guidance,keep_ids,guidance_scale,timesteps,num_warmup_steps,
                    progress_bar,callback,callback_steps,extra_step_kwargs):
        score = None
        # import pdb;pdb.set_trace()
        # expand the latents if we are doing classifier free guidance
        step_timestep: int = t.detach().cpu().item()
        example_latent = all_latents[step_timestep].to(device=self.running_device, dtype=text_embeddings.dtype)
        # example_latent = all_latents[i].to(device=self.running_device, dtype=prompt_embeds.dtype)
        # import pdb; pdb.set_trace()
        latent_list: List[torch.Tensor] = [latents, example_latent, latents]
        latent_model_input: torch.Tensor = torch.cat(latent_list, dim=0).to('cuda') 
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).detach() # detach() aligns with freecontrol
        self.latent_model_input = latent_model_input
        # del example_latent
        
        step_prompt_embeds_list = [text_embeddings.chunk(2)[0]] * 2 + [ref_prompt_embeds] + [text_embeddings.chunk(2)[1]] * 2
        step_prompt_embeds = torch.cat(step_prompt_embeds_list, dim=0).to('cuda')

        down_block_additional_residuals = mid_block_additional_residual = None
        
        require_grad_flag = False
        
        if _in_step(self.guidance_config.pca_guidance, i):
            require_grad_flag = True

        # print(torch.cuda.memory_allocated()/ (1024**3))
        # import pdb; pdb.set_trace()
        
        # Only require grad when need to compute the gradient for guidance
        if require_grad_flag:
            latent_model_input.requires_grad_(True)
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                noise_pred_example = self.unet(
                        latent_model_input[2].unsqueeze(0), t, 
                        encoder_hidden_states=step_prompt_embeds[2].unsqueeze(0),
                        down_block_additional_residuals = down_block_additional_residuals,
                        mid_block_additional_residual   = mid_block_additional_residual,
                    ).sample.to(dtype=latents_dtype)
                if _in_step(self.guidance_config.cross_attn, i):
                    # Compute the Cross-Attention loss and update the cross attention mask, Please don't delete this
                    self.compute_cross_attn_mask(cond_control_ids, cond_example_ids, cond_appearance_ids)
                key_example = self.get_attn_pca_key()  
                
                temp_attn_prob_example = self.get_temp_attn_prob()

                # import pdb; pdb.set_trace()
                
                noise_pred_no_grad =  self.unet(
                        latent_model_input[[0,1,4]], t, 
                        encoder_hidden_states=step_prompt_embeds[[0,1,4]],
                        down_block_additional_residuals = down_block_additional_residuals,
                        mid_block_additional_residual   = mid_block_additional_residual,
                    ).sample.to(dtype=latents_dtype) 
                key_no_grad = self.get_attn_pca_key()

                temp_attn_prob_no_grad = self.get_temp_attn_prob()


                #  这里的noise_pred_example 和 noise_pred_no_grad 后面可以合并并行
            noise_pred_control = self.unet(
                        latent_model_input[3].unsqueeze(0), t, 
                        encoder_hidden_states=step_prompt_embeds[3].unsqueeze(0),
                        down_block_additional_residuals = down_block_additional_residuals,
                        mid_block_additional_residual   = mid_block_additional_residual,
                    ).sample.to(dtype=latents_dtype)
            key_control = self.get_attn_pca_key()

            temp_attn_prob_control = self.get_temp_attn_prob()


            noise_pred = torch.cat([noise_pred_no_grad[[0,1]],noise_pred_example,noise_pred_control,noise_pred_no_grad[2].unsqueeze(0)],dim=0)
            key_list = [key_example, key_control, key_no_grad]

            temp_attn_prob_list = [temp_attn_prob_example, temp_attn_prob_control, temp_attn_prob_no_grad]

            # if i == 0 or i == 100 or i == 199: 
            #     self.temp_attn_prob_dic[i] = temp_attn_prob_example
            # import pdb; pdb.set_trace()
            # key_used = torch.cat([key_example,key_control,key_no_grad[2].unsqueeze(0)],dim=0)
            # key_used = key_used.reshape(-1, key_used.shape[2], key_used.shape[3])
            # import pdb; pdb.set_trace()
            # print("test")
            

        else:
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, 
                    encoder_hidden_states=step_prompt_embeds,
                    down_block_additional_residuals = down_block_additional_residuals,
                    mid_block_additional_residual   = mid_block_additional_residual,
                ).sample.to(dtype=latents_dtype)

                temp_attn_prob_example = self.get_temp_attn_prob()
            


        if i == 0 or i == 100 or i == 199: 
            self.temp_attn_prob_dic[i] = temp_attn_prob_example

        loss = 0
        self.cross_seg = None
        # torch.cuda.empty_cache()
        
        # print(torch.cuda.memory_allocated()/ (1024**3))
        # import pdb; pdb.set_trace()
        
        
        if _in_step(self.guidance_config.pca_guidance, i):
            # Compute the PCA structure and appearance guidance
            # Set the select feature to key by default
            try:
                select_feature = self.guidance_config.pca_guidance.select_feature
            except:
                select_feature = "key"

            if select_feature == 'query' or select_feature == 'key' or select_feature == 'value':
                # calcaulate pca loss
                pca_loss = self.compute_attn_pca_loss(key_list, temp_attn_prob_list, cond_control_ids, cond_example_ids, cond_appearance_ids, i) 
                
                loss += pca_loss
                
            elif select_feature == 'conv':
                pca_loss = self.compute_conv_pca_loss(key_list, temp_attn_prob_list, cond_control_ids, cond_example_ids, cond_appearance_ids, i)
                
                loss += pca_loss
        
        # print(torch.cuda.memory_allocated()/ (1024**3))
        # import pdb; pdb.set_trace()


        temp_control_ids = None
        if isinstance(loss, torch.Tensor):
            gradient = torch.autograd.grad(loss, latent_model_input, allow_unused=True)[0] # [5, 4, 64, 64], 梯度集中在example和control两个latent上面（后者尤其多）
            # import pdb; pdb.set_trace()
            gradient = gradient[cond_control_ids] # [1, 4, 64, 64], 梯度集中在example和control两个latent上面（后者尤其多）

            assert gradient is not None, f"Step {i}: grad is None"
            score = gradient.detach()
            # del gradient
            # del latent_model_input
            # del loss
            temp_control_ids: List[int] = np.arange(num_control_samples).tolist() # [0]
            
        # perform guidance
        if do_classifier_free_guidance:
            # Remove the example samples
            noise_pred = noise_pred[keep_ids] # [5, 4, 64, 64] -> [4, 4, 64, 64]，去掉中间的data_example_latent的noise（condition image的noise）
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # both [2, 4, 64, 64], 前面那个是没有条件指导时预测的噪声
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) # [2, 4, 64, 64]
            # del noise_pred_uncond
            # del noise_pred_text

        guidance_rescale = 0
        if do_classifier_free_guidance and guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale) # [2, 4, 64, 64]
        
        latents = self.scheduler.step(noise_pred, t, latents, score=score,
                                        guidance_scale=self.freecontrol_config.sd_config.grad_guidance_scale,
                                        indices=temp_control_ids,
                                        **extra_step_kwargs, return_dict=False)[0].detach() # [2, 4, 64, 64]
        
        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            progress_bar.update()
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        return latents
    
    def get_attn_pca_key(self,):

        key_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "Attention" in module_name and 'attn1' in name and 'attentions' in name and \
                    _classify_blocks(self.guidance_config.pca_guidance.blocks, name):
                key = module.processor.key
                key = key.reshape(-1, self.input_config.L, key.shape[1], key.shape[2])
                # key = key.reshape(-1, key.shape[2], key.shape[3])
                key_dic[name] = key
                # import pdb; pdb.set_trace()
        return key_dic

    
    def get_cross_attn_prob(self, index_select=None): 
        
        attn_prob_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
                    
            if "CrossAttention" in module_name and 'attn2' in name and 'attentions' in name and _classify_blocks(self.input_config.app_guidance.cross_attn_blocks, name):
                query = module.processor.query
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=query.shape[0]//len(index_select))
                    index_all = torch.arange(query.shape[0])
                    index_picked = index_all[get_index.bool()]
                    query = query[index_picked]
                query = module.reshape_heads_to_batch_dim(query).contiguous()
                # [frames*head, H*W, dim]
                # batch_size * head_size, seq_len, dim // head_size
                
                key = module.processor.key
                if index_select is not None:
                    key = key[index_picked]
                key = module.reshape_heads_to_batch_dim(key).contiguous()
                
                attention_probs = module.get_attention_scores(query, key, None) 
                # [frames*head,H*W, 77]
                attn_prob_dic[name] = attention_probs
                
        return attn_prob_dic
    

    def get_temp_attn_prob(self,index_select=None):

        attn_prob_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "VersatileAttention" in module_name and _classify_blocks(self.input_config.temp_guidance.blocks, name):
                key = module.processor.key
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                key = module.reshape_heads_to_batch_dim(key).contiguous()
                
                query = module.processor.query
                if index_select is not None:
                    query = query[index_picked]
                query = module.reshape_heads_to_batch_dim(query).contiguous()
                

                attention_probs = module.get_attention_scores(query, key, None)         
                attention_probs = attention_probs.reshape(-1, module.heads,attention_probs.shape[1], attention_probs.shape[2])
                
                attn_prob_dic[name] = attention_probs

        return attn_prob_dic
    
    def get_temp_attn_key(self,index_select=None):

        attn_key_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "VersatileAttention" in module_name and _classify_blocks(self.input_config.app_guidance.blocks, name):
                key = module.processor.key
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                
                attn_key_dic[name] = key

        return attn_key_dic
        
    def get_spatial_attn1_key(self, index_select=None):
        attn_key_dic = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "Attention" in module_name and 'attn1' in name and 'attentions' in name and _classify_blocks(self.input_config.app_guidance.blocks, name):
                key = module.processor.key
                # [64,256,1280]
                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                
                attn_key_dic[name] = key
                # [frame, H*W, head*dim] [16,256,1280]

        return attn_key_dic

    
    def __call__(
        self,
        config: omegaconf.dictconfig = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noisy_latents: Optional[torch.FloatTensor] = None,
        inversion_data_path: str = None,
    ):
        # assert config is not None, "config is required for FreeControl pipeline"
        if not hasattr(self, 'config'):
            setattr(self, 'input_config', config)
        self.input_config = config
        if not hasattr(self, 'video_name'):
            setattr(self, 'video_name', config.video_path.split('/')[-1].split('.')[0])
        self.video_name = config.video_path.split('/')[-1].split('.')[0]

        self.unet = prep_unet_attention(self.unet)
        self.unet = prep_unet_conv(self.unet)
        
        # 0. Default height and width to unet
        height = config.height or self.unet.config.sample_size * self.vae_scale_factor
        width = config.width or self.unet.config.sample_size * self.vae_scale_factor
        video_length = config.video_length

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        
        # perform classifier_free_guidance in default
        cfg_scale = config.cfg_scale or 7.5
        do_classifier_free_guidance = True
        
        # 3. Encode input prompt
        new_prompt = config.new_prompt if isinstance(config.new_prompt, list) else [config.new_prompt] * batch_size
        negative_prompt = config.negative_prompt or ""
        negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(new_prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt)
        # [uncond_embeddings, text_embeddings] [2, 77, 768]
        
        if config.get('obj_pairs') is None or len(config.obj_pairs) == 0:
            global_app_guidance = True 
            token_index_example, token_index_app = None, None
        else:
            if len(config.obj_pairs) != 2:
                raise ValueError("only support single object in both original prompt and new prompt")
            global_app_guidance = False
            token_index_example = get_object_index(self.tokenizer,config.inversion_prompt, config.obj_pairs[0])
            token_index_app = get_object_index(self.tokenizer,config.new_prompt, config.obj_pairs[1])
            
            print("token_index_example:", token_index_example, "token_index_app:", token_index_app)
            
        num_inference_step = config.num_inference_step or 300
        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        control_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            noisy_latents,
        )
        latents_group = torch.cat([control_latents, control_latents],dim=0)
        latents_group_app = latents_group
        
        with open(inversion_data_path, "rb") as f:
            inverted_data = pickle.load(f)
            all_latents = inverted_data['all_latents_inversion']
            example_prompt_embeds = inverted_data['inversion_prompt_embeds'].to(device)

        self.temp_attn_prob_dic = {} # for record usage

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        # self.guidance_config = self.config.guidance
        weight_each_motion = torch.tensor(config.temp_guidance.weight_each).to(latents_group.dtype).to(device)
        weight_each_motion = torch.repeat_interleave(weight_each_motion/100.0, repeats=6) 
        #  the 100.0 here is only to avoid numberical overflow under float16
        weight_each_app = torch.tensor(config.app_guidance.weight_each).to(latents_group.dtype).to(device)
        if self.input_config.app_guidance.block_type == "temp":
            weight_each_app = torch.repeat_interleave(weight_each_app/100.0, repeats=6) 
        else:
            weight_each_app = torch.repeat_interleave(weight_each_app/100.0, repeats=3)
        
        example_latents = control_latents
        
        with self.progress_bar(total=num_inference_step) as progress_bar:
            for step_index, step_t in enumerate(self.scheduler.timesteps):
                # for i in tqdm(range(num_inv_steps)): step_t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1] 反转时
                global step_idx
                step_idx = step_index
                if step_index <self.input_config.guidance_step:
                    if step_t.item() not in all_latents.keys():
                        raise IndexError("The inference step does not match the inversion step")
                    example_latents = all_latents[step_t.item()].to(device=device, dtype=text_embeddings.dtype)
                latents_group,latents_group_app = self.single_step_video(latents_group, latents_group_app, step_index, step_t, example_latents, text_embeddings, example_prompt_embeds,
                                                cfg_scale, weight_each_motion, weight_each_app, global_app_guidance, token_index_example, token_index_app,extra_step_kwargs)                              
                
                progress_bar.update()
            
            control_latents = latents_group[[1]]
            # 8. Post-processing
            video = self.decode_latents(control_latents)
        return video

    def single_step_video(self, latents_group, latents_group_app, step_index, step_t, example_latents, text_embeddings, 
                            example_prompt_embeds, cfg_scale, weight_each_motion, weight_each_app, global_app_guidance, token_index_example, token_index_app, extra_step_kwargs):
        
        
        # Only require grad when need to compute the gradient for guidance
        if step_index < self.input_config.guidance_step:
            latent_model_input: torch.Tensor = torch.cat([latents_group[[0]], example_latents, latents_group[[1]],latents_group_app[[0]],latents_group_app[[1]]], dim=0)
            step_prompt_embeds = torch.cat([text_embeddings[[0]], example_prompt_embeds, text_embeddings[[1]], text_embeddings[[0]], text_embeddings[[1]]], dim=0)
            # [uncondition_latent, example_latent, control_latent, uncondition_app, condition_app]
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_t).detach() # detach() aligns with freecontrol
        
            latent_model_input.requires_grad = True
            # latent_model_input: [uncondition_latent, example_latent, control_latent, uncondition_app, condition_app]
            with torch.no_grad():
                noise_pred_no_grad = self.unet(
                        latent_model_input[[0,1,3,4]], step_t, 
                        encoder_hidden_states=step_prompt_embeds[[0,1,3,4]],
                    ).sample.to(dtype=latents_group.dtype)
                temp_attn_prob_example = self.get_temp_attn_prob(index_select=[0,1,0,0])
                if self.input_config.app_guidance.block_type =="temp":
                    attn_key_app = self.get_temp_attn_key(index_select=[0,0,0,1])
                else:
                    attn_key_app = self.get_spatial_attn1_key(index_select=[0,0,0,1])
                    
                cross_attn2_prob = self.get_cross_attn_prob(index_select=[0,1,0,1])
                if not global_app_guidance:
                    mask_example_fore, mask_example_back, mask_app_fore, mask_app_back = compute_cross_attn_mask(self.input_config.app_guidance.cross_attn_blocks, cross_attn2_prob, 
                        token_index_example, token_index_app, self.input_config.app_guidance.cross_attn_mask_tr_example,self.input_config.app_guidance.cross_attn_mask_tr_app, step_index=None) 

                    # [frame,  H*W, 1]
            # torch.cuda.empty_cache()
            noise_pred_control = self.unet(
                        latent_model_input[[2]], step_t, 
                        encoder_hidden_states=step_prompt_embeds[[2]],
                    ).sample.to(dtype=latents_group.dtype)
            temp_attn_prob_control = self.get_temp_attn_prob()
            if self.input_config.app_guidance.block_type =="temp":
                attn_key_control = self.get_temp_attn_key()
                # [H*W, frame, head*dim]
            else:
                attn_key_control = self.get_spatial_attn1_key()
                # [frame, H*W, head*dim]
            # cross_attn2_prob_control = self.get_cross_attn_prob()
            # # [frames*head, H*W, 77]
            loss_motion = compute_temp_loss(temp_attn_prob_example, temp_attn_prob_control, weight_each_motion.detach(),None)
            if global_app_guidance:
                loss_appearance = compute_semantic_loss(attn_key_app,attn_key_control, weight_each_app.detach(),
                                                            None, None, None,None,block_type=self.input_config.app_guidance.block_type)
            else:      
                loss_appearance = compute_semantic_loss(attn_key_app,attn_key_control, weight_each_app.detach(),
                                                            mask_example_fore, mask_example_back, mask_app_fore,mask_app_back,block_type=self.input_config.app_guidance.block_type)
            

            ####################################
            if step_index > self.input_config.guidance_step- self.input_config.cool_up_step:
                loss_motion = 0.0 * loss_motion
            ######################################


            loss_total = 100.0*(loss_motion + loss_appearance) 
            
            if step_index < self.input_config.warm_up_step:
                scale = (step_index+1)/self.input_config.warm_up_step
                loss_total = scale*loss_total
                # print(scale.item())
            # if step_index > self.input_config.guidance_step- self.input_config.cool_up_step:
            #     scale = (self.input_config.guidance_step-step_index)/self.input_config.cool_up_step
            #     loss_total = scale*loss_total
                # print(scale.item())
            # loss_motion = loss_motion*self.input_config.temp_guidance.weight
            
            gradient = torch.autograd.grad(loss_total, latent_model_input, allow_unused=True)[0] # [5, 4, 64, 64], 梯度集中在control_latent上
            gradient = gradient[[2]] # [1, 4, 64, 64], 梯度集中control_latents
            assert gradient is not None, f"Step {step_index}: grad is None"
            if self.input_config.grad_guidance_threshold is not None:
                threshold = self.input_config.grad_guidance_threshold
                gradient_clamped = torch.where(
                        gradient.abs() > threshold,
                        torch.sign(gradient) * threshold,
                        gradient
                    )
                score = gradient_clamped.detach()
            else:
                score = gradient.detach()
                
            noise_pred = noise_pred_control + cfg_scale * (noise_pred_control - noise_pred_no_grad[[0]]) # [2, 4, 64, 64]
            noise_app =  noise_pred_no_grad[[3]] + cfg_scale * (noise_pred_no_grad[[3]] - noise_pred_no_grad[[2]]) # [2, 4, 64, 64]
            
            control_latents = self.scheduler.customized_step(noise_pred, step_t, latents_group[[1]], score=score,
                                        guidance_scale=self.input_config.grad_guidance_scale,
                                        indices=[0],
                                        **extra_step_kwargs, return_dict=False)[0].detach() # [1, 4, 64, 64]
            app_latents = self.scheduler.customized_step(noise_app, step_t, latents_group_app[[1]], score=None,
                                            guidance_scale=self.input_config.grad_guidance_scale,
                                            indices=[0],
                                            **extra_step_kwargs, return_dict=False)[0].detach() # [1, 4, 64, 64]
            return torch.cat([control_latents , control_latents],dim=0), torch.cat([app_latents , app_latents],dim=0)
            
        else:
            with torch.no_grad():
                latent_model_input = self.scheduler.scale_model_input(latents_group, step_t) # detach() aligns with freecontrol
                noise_pred = self.unet(
                    latent_model_input, step_t, 
                    encoder_hidden_states=text_embeddings,
                ).sample.to(dtype=latents_group.dtype)

                noise_pred = noise_pred[[1]] + cfg_scale * (noise_pred[[1]] - noise_pred[[0]])
                control_latents = self.scheduler.customized_step(noise_pred, step_t, latents_group[[1]], score=None,
                                                guidance_scale=self.input_config.grad_guidance_scale,
                                                indices=[0],
                                                **extra_step_kwargs, return_dict=False)[0] # [1, 4, 64, 64]
                return torch.cat([control_latents , control_latents],dim=0), None
