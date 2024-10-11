from dataclasses import dataclass
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Callable, List, Optional, Union
from diffusers.utils import deprecate, logging, BaseOutput
from .xformer_attention import *
from .conv_layer import *
from .util import *
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
from motionclone.utils.util import video_preprocess
import einops
import torchvision.transforms as transforms

def add_noise(self, timestep, x_0, noise_pred):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        latents_noise = alpha_prod_t ** 0.5 * x_0 + beta_prod_t ** 0.5 * noise_pred
        return latents_noise
    
@torch.no_grad()
def obtain_motion_representation(self, generator=None, motion_representation_path: str = None,
                                 duration=None,use_controlnet=False,):
    
    video_data = video_preprocess(self.input_config.video_path, self.input_config.height, 
                                  self.input_config.width, self.input_config.video_length,duration=duration)
    video_latents = self.vae.encode(video_data.to(self.vae.dtype).to(self.vae.device)).latent_dist.sample(None)
    video_latents = self.vae.config.scaling_factor * video_latents
    video_latents = video_latents.unsqueeze(0)
    video_latents = einops.rearrange(video_latents, "b f c h w -> b c f h w")
    
    uncond_input = self.tokenizer(
        [""], padding="max_length", max_length=self.tokenizer.model_max_length,
        return_tensors="pt"
    )
    step_t = int(self.input_config.add_noise_step)
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    noise_sampled = randn_tensor(video_latents.shape, generator=generator, device=video_latents.device, dtype=video_latents.dtype)
    noisy_latents = self.add_noise(step_t, video_latents, noise_sampled)
            
    down_block_additional_residuals = mid_block_additional_residual = None
    if use_controlnet:
        controlnet_image_index = self.input_config.image_index 
        if self.controlnet.use_simplified_condition_embedding:
            controlnet_images = video_latents[:,:,controlnet_image_index,:,:] 
        else:
            controlnet_images = (einops.rearrange(video_data.unsqueeze(0).to(self.vae.dtype).to(self.vae.device), "b f c h w -> b c f h w")+1)/2
            controlnet_images = controlnet_images[:,:,controlnet_image_index,:,:]

        controlnet_cond_shape    = list(controlnet_images.shape)
        controlnet_cond_shape[2] = noisy_latents.shape[2]
        controlnet_cond = torch.zeros(controlnet_cond_shape).to(noisy_latents.device).to(noisy_latents.dtype)

        controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
        controlnet_conditioning_mask_shape[1] = 1
        controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(noisy_latents.device).to(noisy_latents.dtype)

        controlnet_cond[:,:,controlnet_image_index] = controlnet_images
        controlnet_conditioning_mask[:,:,controlnet_image_index] = 1

        down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
            noisy_latents, step_t,
            encoder_hidden_states=uncond_embeddings,
            controlnet_cond=controlnet_cond,
            conditioning_mask=controlnet_conditioning_mask,
            conditioning_scale=self.input_config.controlnet_scale,
            guess_mode=False, return_dict=False,
        )

    _ = self.unet(noisy_latents, step_t, encoder_hidden_states=uncond_embeddings, return_dict=False, only_motion_feature=True,
                  down_block_additional_residuals = down_block_additional_residuals,
                  mid_block_additional_residual = mid_block_additional_residual,)
    temp_attn_prob_control = self.get_temp_attn_prob()
   
    motion_representation = { key: [max_value, max_index.to(torch.uint8)] for key, tensor in temp_attn_prob_control.items() for max_value, max_index in [torch.topk(tensor, k=1, dim=-1)]} 
    
    torch.save(motion_representation, motion_representation_path)
    self.motion_representation_path = motion_representation_path


def compute_temp_loss(self, temp_attn_prob_control_dict):
    temp_attn_prob_loss = []
    for name in temp_attn_prob_control_dict.keys():
        current_temp_attn_prob = temp_attn_prob_control_dict[name]
        reference_representation_dict = self.motion_representation_dict[name]

        max_index = reference_representation_dict[1].to(torch.int64).to(current_temp_attn_prob.device)
        current_motion_representation = torch.gather(current_temp_attn_prob, index = max_index, dim=-1)
        
        reference_motion_representation  = reference_representation_dict[0].to(dtype = current_motion_representation.dtype, device = current_motion_representation.device)
        
        module_attn_loss = F.mse_loss(current_motion_representation,  reference_motion_representation.detach())
        temp_attn_prob_loss.append(module_attn_loss)
    
    loss_temp = torch.stack(temp_attn_prob_loss)
    return loss_temp.sum()
     
def sample_video(
    self,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    noisy_latents: Optional[torch.FloatTensor] = None,
    add_controlnet: bool = False,
):
    # Determine if use controlnet, i.e., conditional image2video
    self.add_controlnet = add_controlnet
    if self.add_controlnet:
        image_transforms = transforms.Compose([
            transforms.Resize((self.input_config.height, self.input_config.width)),
            transforms.ToTensor(),
        ])
            
        controlnet_images = [image_transforms(Image.open(path).convert("RGB")) for path in self.input_config.condition_image_path_list]
        controlnet_images = torch.stack(controlnet_images).unsqueeze(0).to(dtype=self.vae.dtype,device=self.vae.device)
        controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

        with torch.no_grad():
            if self.controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = self.vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * self.vae.config.scaling_factor
                self.controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)
            else:
                self.controlnet_images = controlnet_images
        

    # Define call parameters
    # perform classifier_free_guidance in default
    batch_size = 1
    do_classifier_free_guidance = True
    device = self._execution_device
    
    # Encode input prompt
    self.text_embeddings = self._encode_prompt(self.input_config.new_prompt, device, 1, do_classifier_free_guidance, self.input_config.negative_prompt)
    # [uncond_embeddings, text_embeddings] [2, 77, 768]
    
    # Prepare latent variables
    noisy_latents = self.prepare_latents(
        batch_size,
        self.unet.config.in_channels,
        self.input_config.video_length,
        self.input_config.height,
        self.input_config.width,
        self.text_embeddings.dtype,
        device,
        generator,
        noisy_latents,
    )
    
    self.motion_representation_dict = torch.load(self.motion_representation_path)
    self.motion_scale = self.input_config.motion_guidance_weight

    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    
    # save GPU memory
    # self.vae.to(device = "cpu")
    # self.text_encoder.to(device = "cpu")
    # torch.cuda.empty_cache()

    with self.progress_bar(total=self.input_config.inference_steps) as progress_bar:
        for step_index, step_t in enumerate(self.scheduler.timesteps):
            noisy_latents = self.single_step_video(noisy_latents, step_index, step_t, extra_step_kwargs)                              
            progress_bar.update()
        
        # decode latents for videos
        video = self.decode_latents(noisy_latents)
    return video

def single_step_video(self, noisy_latents, step_index, step_t,  extra_step_kwargs):

    down_block_additional_residuals = mid_block_additional_residual = None
    if self.add_controlnet:
        with torch.no_grad(): 
            controlnet_cond_shape    = list(self.controlnet_images.shape)
            controlnet_cond_shape[2] = noisy_latents.shape[2]
            controlnet_cond = torch.zeros(controlnet_cond_shape).to(noisy_latents.device).to(noisy_latents.dtype)

            controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
            controlnet_conditioning_mask_shape[1] = 1
            controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(noisy_latents.device).to(noisy_latents.dtype)

            controlnet_image_index = self.input_config.image_index
            controlnet_cond[:,:,controlnet_image_index] = self.controlnet_images
            controlnet_conditioning_mask[:,:,controlnet_image_index] = 1

            down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                noisy_latents.expand(2,-1,-1,-1,-1), step_t,
                encoder_hidden_states=self.text_embeddings,
                controlnet_cond=controlnet_cond,
                conditioning_mask=controlnet_conditioning_mask,
                conditioning_scale=self.input_config.controlnet_scale,
                guess_mode=False, return_dict=False,
            )

    # Only require grad when need to compute the gradient for guidance
    if step_index < self.input_config.guidance_steps:
    
        down_block_additional_residuals_uncond = down_block_additional_residuals_cond = None
        mid_block_additional_residual_uncond = mid_block_additional_residual_cond = None
        if self.add_controlnet:
            down_block_additional_residuals_uncond = [tensor[[0],...].detach() for tensor in down_block_additional_residuals]
            down_block_additional_residuals_cond = [tensor[[1],...].detach() for tensor in down_block_additional_residuals]
            mid_block_additional_residual_uncond =  mid_block_additional_residual[[0],...].detach()
            mid_block_additional_residual_cond = mid_block_additional_residual[[1],...].detach()
          
        control_latents = noisy_latents.clone().detach()
        control_latents.requires_grad = True

        control_latents = self.scheduler.scale_model_input(control_latents, step_t) 
        noisy_latents = self.scheduler.scale_model_input(noisy_latents, step_t)
    
        with torch.no_grad():
            noise_pred_uncondition = self.unet(noisy_latents, step_t, encoder_hidden_states=self.text_embeddings[[0]],
                                        down_block_additional_residuals = down_block_additional_residuals_uncond,
                                        mid_block_additional_residual = mid_block_additional_residual_uncond,).sample.to(dtype=noisy_latents.dtype)

        noise_pred_condition = self.unet(control_latents, step_t, encoder_hidden_states=self.text_embeddings[[1]],
                                         down_block_additional_residuals = down_block_additional_residuals_cond,
                                         mid_block_additional_residual = mid_block_additional_residual_cond,).sample.to(dtype=noisy_latents.dtype)
        temp_attn_prob_control = self.get_temp_attn_prob()
        
        loss_motion = self.motion_scale * self.compute_temp_loss(temp_attn_prob_control,)
        
        if step_index < self.input_config.warm_up_steps:
            scale = (step_index+1)/self.input_config.warm_up_steps
            loss_motion = scale*loss_motion
        
        if step_index > self.input_config.guidance_steps-self.input_config.cool_up_steps:
            scale = (self.input_config.guidance_steps-step_index)/self.input_config.cool_up_steps
            loss_motion = scale*loss_motion

        gradient = torch.autograd.grad(loss_motion, control_latents, allow_unused=True)[0] # [1, 4, 16, 64, 64],
        assert gradient is not None, f"Step {step_index}: grad is None"

        noise_pred = noise_pred_condition + self.input_config.cfg_scale * (noise_pred_condition - noise_pred_uncondition)
        
        control_latents = self.scheduler.customized_step(noise_pred, step_index, control_latents, score=gradient.detach(),
                                        **extra_step_kwargs, return_dict=False)[0] # [1, 4, 16, 64, 64]
        return control_latents.detach()

    else:
        with torch.no_grad():
            noisy_latents = self.scheduler.scale_model_input(noisy_latents, step_t) 
            noise_pred_group = self.unet(
                noisy_latents.expand(2,-1,-1,-1,-1), step_t, 
                encoder_hidden_states=self.text_embeddings,
                down_block_additional_residuals = down_block_additional_residuals,
                mid_block_additional_residual = mid_block_additional_residual,
            ).sample.to(dtype=noisy_latents.dtype)

            noise_pred = noise_pred_group[[1]] + self.input_config.cfg_scale * (noise_pred_group[[1]] - noise_pred_group[[0]])
            noisy_latents = self.scheduler.customized_step(noise_pred, step_index, noisy_latents, score=None, **extra_step_kwargs, return_dict=False)[0] # [1, 4, 16, 64, 64]
        return noisy_latents.detach()
  

def get_temp_attn_prob(self,index_select=None):
        
        attn_prob_dic = {}

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "VersatileAttention" in module_name and classify_blocks(self.input_config.motion_guidance_blocks, name):
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

@torch.no_grad()
def schedule_customized_step(
        self,
        model_output: torch.FloatTensor,
        step_index: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,

        # Guidance parameters
        score=None,
        guidance_scale=1.0,
        indices=None, # [0]
        return_middle = False,
):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    
    # Support IF models
    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None
    
    timestep = self.timesteps[step_index]
    # 1. get previous step value (=t-1)
    # prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    prev_timestep = self.timesteps[step_index+1] if step_index +1 <len(self.timesteps) else -1

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5) # [2, 4, 64, 64]
    
    if score is not None and return_middle:
        return pred_epsilon, alpha_prod_t, alpha_prod_t_prev, pred_original_sample

    # 6. apply guidance following the formula (14) from https://arxiv.org/pdf/2105.05233.pdf
    if score is not None and guidance_scale > 0.0: 
        if indices is not None:
            # import pdb; pdb.set_trace()
            assert pred_epsilon[indices].shape == score.shape, "pred_epsilon[indices].shape != score.shape"
            pred_epsilon[indices] = pred_epsilon[indices] - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score 
        else:
            assert pred_epsilon.shape == score.shape
            pred_epsilon = pred_epsilon - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score
    # 

    # 7. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon 

    # 8. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction 

    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                " `variance_noise` stays `None`."
            )

        if variance_noise is None:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        variance = std_dev_t * variance_noise 

        prev_sample = prev_sample + variance 

    if not return_dict:
        return (prev_sample,)

    return prev_sample, pred_original_sample, alpha_prod_t_prev



def schedule_set_timesteps(self, num_inference_steps: int, guidance_steps: int = 0, guiduance_scale: float = 0.0, device: Union[str, torch.device] = None,timestep_spacing_type= "uneven"):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps
        
        # assign more steps in early denoising stage for motion guidance
        if timestep_spacing_type == "uneven":
            timesteps_guidance = (
                np.linspace(int((1-guiduance_scale)*self.config.num_train_timesteps), self.config.num_train_timesteps - 1, guidance_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
            timesteps_vanilla = (
                np.linspace(0, int((1-guiduance_scale)*self.config.num_train_timesteps) - 1, num_inference_steps-guidance_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
            timesteps = np.concatenate((timesteps_guidance, timesteps_vanilla))
 
        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        elif timestep_spacing_type == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif timestep_spacing_type == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif timestep_spacing_type == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{timestep_spacing_type} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device)
       
@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor

def unet_customized_forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,

        # support controlnet
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,

        return_dict: bool = True,
        only_motion_feature: bool = False,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

            down_block_res_samples += res_samples

        # support controlnet
        down_block_res_samples = list(down_block_res_samples)
        if down_block_additional_residuals is not None:
            for i, down_block_additional_residual in enumerate(down_block_additional_residuals):
                if down_block_additional_residual.dim() == 4: # boardcast
                    down_block_additional_residual = down_block_additional_residual.unsqueeze(2)
                down_block_res_samples[i] = down_block_res_samples[i] + down_block_additional_residual

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        # support controlnet
        if mid_block_additional_residual is not None:
            if mid_block_additional_residual.dim() == 4: # boardcast
                mid_block_additional_residual = mid_block_additional_residual.unsqueeze(2)
            sample = sample + mid_block_additional_residual
        
        # up
        for i, upsample_block in enumerate(self.up_blocks):
            if i<= int(self.input_config.motion_guidance_blocks[-1].split(".")[-1]):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size, encoder_hidden_states=encoder_hidden_states,
                    )
            else:
                if only_motion_feature:
                    return 0
                with torch.no_grad():
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size, encoder_hidden_states=encoder_hidden_states,
                        )
                        
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

