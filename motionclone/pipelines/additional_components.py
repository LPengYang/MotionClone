from dataclasses import dataclass
import os
import pickle
import numpy as np
import torch
import omegaconf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Callable, List, Optional, Union, Any, Dict
from diffusers.utils import deprecate, logging, BaseOutput
from ..utils.xformer_attention import *
from ..utils.conv_layer import *
from ..utils.util import *

@torch.no_grad()
def customized_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,

        # Guidance parameters
        score=None,
        guidance_scale=0.0,
        indices=None, 

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

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

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

    # 6. apply guidance following the formula (14) from https://arxiv.org/pdf/2105.05233.pdf
    if score is not None and guidance_scale > 0.0: 
        if indices is not None:
            # import pdb; pdb.set_trace()
            assert pred_epsilon[indices].shape == score.shape, "pred_epsilon[indices].shape != score.shape"
            pred_epsilon[indices] = pred_epsilon[indices] - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score 
        else:
            assert pred_epsilon.shape == score.shape
            pred_epsilon = pred_epsilon - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score

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
    self.pred_epsilon = pred_epsilon
    if not return_dict:
        return (prev_sample,)

    return prev_sample, pred_original_sample

def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None,timestep_spacing_type= "linspace"):
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

    # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
    if timestep_spacing_type == "linspace":
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


def find_centroid(mask):
    B, H, W = mask.shape
    y_coords = torch.arange(H, dtype=torch.float32, device=mask.device).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, dtype=torch.float32, device=mask.device).view(1, 1, W).expand(B, H, W)
    
    total_mass = mask.sum(dim=(1, 2))
    y_center = (y_coords * mask).sum(dim=(1, 2)) / total_mass
    x_center = (x_coords * mask).sum(dim=(1, 2)) / total_mass
    
    return torch.stack([y_center, x_center], dim=1)

def get_gaussian_blur(mask, sigma=5):
    mask = mask.reshape(mask.shape[0],int(mask.shape[1]**0.5),int(mask.shape[1]**0.5))
    B, H, W = mask.shape
    centroids = find_centroid(mask)
    
    y_coords = torch.arange(H, dtype=torch.float32, device=mask.device).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, dtype=torch.float32, device=mask.device).view(1, 1, W).expand(B, H, W)
    
    y_centers = centroids[:, 0].view(B, 1, 1).expand(B, H, W)
    x_centers = centroids[:, 1].view(B, 1, 1).expand(B, H, W)
    
    gauss_y = torch.exp(-((y_coords - y_centers) ** 2) / (2 * sigma ** 2))
    gauss_x = torch.exp(-((x_coords - x_centers) ** 2) / (2 * sigma ** 2))
    gauss = gauss_y * gauss_x
    
    return gauss.reshape(mask.shape[0],-1,1)

def compute_cross_attn_mask(mask_blocks, cross_attn2_prob,token_index_example, token_index_app, mask_threshold_example=0.2,mask_threshold_app=0.3,step_index=None):
    mask_example_foreground,mask_example_background,mask_app_foreground,mask_app_background  = {}, {}, {}, {}
    
    for block_name in mask_blocks:
        if block_name != "up_blocks.1":
            # [frame, H*W, 1]
            feature = mask_example_foreground["up_blocks.1"]
            shape = feature.shape
            res = int(np.sqrt(feature.shape[1]))
            # import pdb; pdb.set_trace()
            feature = F.interpolate(feature.permute(0,2,1).reshape(shape[0],1,res,res),(2 * res, 2 * res),mode='nearest')
            mask_example_foreground[block_name]  = feature.reshape(shape[0],1,-1).permute(0,2,1)  
            
            feature = mask_example_background["up_blocks.1"]
            shape = feature.shape
            res = int(np.sqrt(feature.shape[1]))
            feature = F.interpolate(feature.permute(0,2,1).reshape(shape[0],1,res,res),(2 * res, 2 * res),mode='nearest')
            mask_example_background[block_name]  = feature.reshape(shape[0],1,-1).permute(0,2,1)
            
            feature = mask_app_foreground["up_blocks.1"]
            shape = feature.shape
            res = int(np.sqrt(feature.shape[1]))
            feature = F.interpolate(feature.permute(0,2,1).reshape(shape[0],1,res,res),(2 * res, 2 * res),mode='nearest')
            mask_app_foreground[block_name]  = feature.reshape(shape[0],1,-1).permute(0,2,1)
            
            feature = mask_app_background["up_blocks.1"]
            shape = feature.shape
            res = int(np.sqrt(feature.shape[1]))
            feature = F.interpolate(feature.permute(0,2,1).reshape(shape[0],1,res,res),(2 * res, 2 * res),mode='nearest')
            mask_app_background[block_name]  = feature.reshape(shape[0],1,-1).permute(0,2,1)
                                                        
        else:
            attn2_prob_example = []
            attn2_prob_app = []
            corss_attn2_prob_each_block = {key: cross_attn2_prob[key] for key in cross_attn2_prob.keys() if block_name in key}
            
            for name in corss_attn2_prob_each_block.keys():
                cross_attn2_prob_each = corss_attn2_prob_each_block[name]
                cross_attn2_prob_each = cross_attn2_prob_each.reshape(2,16,-1,cross_attn2_prob_each.shape[1],cross_attn2_prob_each.shape[2])
                attn2_prob_example.append(cross_attn2_prob_each[0])
                attn2_prob_app.append(cross_attn2_prob_each[1])
                # [16, head, H*W, 77]
            
            attn2_prob_example = torch.mean(torch.cat(attn2_prob_example, dim = 1),dim=1)
            attn2_prob_app = torch.mean(torch.cat(attn2_prob_app, dim = 1),dim=1)
            
            # [frame,H*W,1]
            mask_example = attn2_prob_example[:,:,[token_index_example]]
            mask_example = (mask_example - mask_example.min(dim=1,keepdim=True)[0])/(mask_example.max(dim=1,keepdim=True)[0]-mask_example.min(dim=1,keepdim=True)[0]+1e-5)


            ######################### 
            # mask_example_foreground[block_name] = (mask_example > mask_threshold_example).to(attn2_prob_example.dtype)
            # mask_example_background[block_name] = 1-mask_example_foreground[block_name]
            #########################
            
            mask_no_blur = (mask_example>mask_threshold_example).to(attn2_prob_example.dtype)
            gaussian_blur = get_gaussian_blur(mask_no_blur,sigma=5).to(mask_no_blur.dtype)
            mask_example_foreground[block_name] = gaussian_blur* mask_no_blur
            mask_example_background[block_name] = (1-gaussian_blur)*(1-mask_no_blur) 
            
            
            mask_app = attn2_prob_app[:,:,[token_index_app]]
            mask_app = (mask_app - mask_app.min(dim=1,keepdim=True)[0])/(mask_app.max(dim=1,keepdim=True)[0]-mask_app.min(dim=1,keepdim=True)[0]+1e-5)

            mask_app_foreground[block_name] = (mask_app>mask_threshold_app).to(attn2_prob_app.dtype)
            mask_app_background[block_name] = 1-mask_app_foreground[block_name]
                
        if step_index is not None and step_index % 30 ==0:
            for index in range(mask_example_foreground[block_name].shape[0]):
                mask_example_each = mask_example_foreground[block_name][index]
                res = int(np.sqrt(mask_example_each.shape[0]))
                mask_example_each = mask_example_each.reshape(res,res).cpu().numpy() * 255
                mask_example_each =  Image.fromarray(mask_example_each.astype(np.uint8))
                save_path = os.path.join("masks","example_"+ block_name +"_" +str(step_index) +"_" +str(index)+".png")
                mask_example_each.save(save_path)
                
                mask_app_each = mask_app_foreground[block_name][index]
                mask_app_each = mask_app_each.reshape(res,res).cpu().numpy() * 255
                mask_app_each =  Image.fromarray(mask_app_each.astype(np.uint8))
                save_path = os.path.join("masks","app_"+ block_name +"_" +str(step_index) +"_" +str(index)+".png")
                mask_app_each.save(save_path)
    
    return mask_example_foreground, mask_example_background, mask_app_foreground, mask_app_background

def compute_temp_loss(temp_attn_prob_example, temp_attn_prob_control, weight_each,mask_example):

    temp_attn_prob_loss = []
    # 1. Loop though all layers to get the query, key, and Compute the PCA loss
    for name in temp_attn_prob_example.keys():

        ######################### for visualize
        global block_num
        global module_num
        global layer_num
        block_num, module_num, layer_num = int(name.split(".")[1]), int(name.split(".")[3]), int(name.split(".")[-1])
        #########################

        attn_prob_example = temp_attn_prob_example[name]
        attn_prob_control = temp_attn_prob_control[name]
    
        
        module_attn_loss = calculate_motion_rank(attn_prob_example.detach(), attn_prob_control, rank_k = 1)



        temp_attn_prob_loss.append(module_attn_loss)
            
    loss_temp = torch.stack(temp_attn_prob_loss) * weight_each
    return loss_temp.mean()


def calculate_motion_rank(tensor_ref, tensor_gen, rank_k = 1, use_soft_mse=False, use_weight=False):
    if rank_k ==0:
        loss = torch.tensor(0.0,device = tensor_ref.device)
    elif rank_k >tensor_ref.shape[-1]:
        raise ValueError("the value of rank_k cannot larger than the number of frames")
    else:   
        _, sorted_indices = torch.sort(tensor_ref, dim=-1)
        #  [*tensor_ref.shape[:-1],16-rank_k]
        mask_indices = torch.cat((torch.zeros([*tensor_ref.shape[:-1],tensor_ref.shape[-1]-rank_k], dtype=torch.bool), torch.ones([*tensor_ref.shape[:-1],rank_k], dtype=torch.bool)),dim=-1)
        max_copy = sorted_indices[:,:,:,[-1]].expand(*tensor_ref.shape[:-1], tensor_ref.shape[-1])
        sorted_indices[~mask_indices] = max_copy[~mask_indices]
        mask = torch.zeros_like(tensor_ref, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, True) 
        if use_soft_mse:
            if use_weight:
                weight = calculate_weight(tensor_ref)
                loss = soft_mse(weight[mask]*tensor_ref[mask].detach(),weight[mask]*tensor_gen[mask])
            else:
                loss = soft_mse(tensor_ref[mask].detach(),tensor_gen[mask])
        else:
            if use_weight:
                weight = calculate_weight(tensor_ref)
                loss = F.mse_loss(weight[mask]*tensor_ref[mask].detach(),weight[mask]*tensor_gen[mask])
            else:
                loss = F.mse_loss(tensor_ref[mask].detach(),tensor_gen[mask])
    return loss


def compute_semantic_loss( temp_attn_key_app, temp_attn_key_control, weight_each,mask_example_fore, mask_example_back,mask_app_fore, mask_app_back, block_type="temp"):
    temp_attn_key_loss = []
    for name in temp_attn_key_app.keys():

        attn_key_app = temp_attn_key_app[name]
        attn_key_control = temp_attn_key_control[name]
        if mask_example_fore == None:
            module_attn_loss = calculate_semantic_loss(attn_key_app.detach(), attn_key_control, 
                                                    None, None, None, None,block_type)
        else:
            block_name = ".".join(name.split(".")[:2])
            module_attn_loss = calculate_semantic_loss(attn_key_app.detach(), attn_key_control, 
                                                        mask_example_fore[block_name], mask_example_back[block_name],mask_app_fore[block_name], mask_app_back[block_name],block_type)

        temp_attn_key_loss.append(module_attn_loss)
            
    loss_app = torch.stack(temp_attn_key_loss) * weight_each
    return loss_app.mean()


def calculate_semantic_loss(tensor_app, tensor_gen, mask_example_fore, mask_example_back,mask_app_fore, mask_app_back, block_type):
 
    if mask_example_fore is None:
        if block_type == "temp":
            # loss = F.mse_loss(tensor_app.mean(dim=0).detach(), tensor_gen.mean(dim=0))
            loss = F.mse_loss(tensor_app.mean(dim=[0,1],keepdim=True).detach(), tensor_gen.mean(dim=0,keepdim=True))  
        else:
            # loss = F.mse_loss(tensor_app.mean(dim=1).detach(), tensor_gen.mean(dim=1))  
            loss = F.mse_loss(tensor_app.mean(dim=[0,1], keepdim=True).detach(), tensor_gen.mean(dim=1,keepdim=True))  
 
    else:
        if block_type == "temp":
            tensor_app = tensor_app.permute(1,0,2)
            tensor_gen = tensor_gen.permute(1,0,2)
        # [frame, H*W, head*dim]

        ref_foreground = (tensor_app*mask_app_fore).sum(dim=1)/(mask_app_fore.sum(dim=1)+1e-5)
        ref_background = (tensor_app*mask_app_back).sum(dim=1)/(mask_app_back.sum(dim=1)+1e-5)
        
        gen_foreground = (tensor_gen*mask_example_fore).sum(dim=1)/(mask_example_fore.sum(dim=1)+1e-5)
        gen_background = (tensor_gen*mask_example_back).sum(dim=1)/(mask_example_back.sum(dim=1)+1e-5)
        
        loss = F.mse_loss(ref_foreground.detach(), gen_foreground) + F.mse_loss(ref_background.detach(), gen_background) 

    return loss

def get_object_index(tokenizer, prompt: str, word: str,):
    tokens_list = tokenizer(prompt.lower()).input_ids
    search_tokens = tokenizer(word.lower()).input_ids
    token_index = tokens_list.index(search_tokens[1])
    return token_index