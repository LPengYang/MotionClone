import math
from typing import Optional, Callable

import xformers
from omegaconf import OmegaConf
import yaml


def identify_blocks(block_list, name):
    block_name = None
    for block in block_list:
        if block in name:
            block_name = block
            break
    return block_name


def classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block


class MySelfAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
        

    def __call__(self, attn, hidden_states, query, key, value, attention_mask):
        self.attn = attn
        self.key = key
        self.query = query
        self.value = value
        self.attention_mask = attention_mask
        self.hidden_state = hidden_states.detach()
        return hidden_states
    
    def record_qkv(self, attn, hidden_states, query, key, value, attention_mask):
        self.key = key
        self.query = query
        self.value = value
        self.hidden_state = hidden_states.detach()
        
    def record_attn_mask(self, attn, hidden_states, query, key, value, attention_mask):
        self.attn = attn
        self.attention_mask = attention_mask
        

def prep_unet_attention(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if "Attention" in module_name:
            module.set_processor(MySelfAttnProcessor())
    return unet


def get_self_attn_feat(unet, injection_config, config):
    hidden_state_dict = dict()
    query_dict = dict()
    key_dict = dict()
    value_dict = dict()
    
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if "CrossAttention" in module_name and 'attn1' in name and classify_blocks(injection_config.blocks, name=name):
            res = int(math.sqrt(module.processor.hidden_state.shape[1]))
            bs = module.processor.hidden_state.shape[0] # 20 * 16 = 320
            hidden_state_dict[name] = module.processor.hidden_state.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            res = int(math.sqrt(module.processor.query.shape[1]))
            query_dict[name] = module.processor.query.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            key_dict[name] = module.processor.key.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            value_dict[name] = module.processor.value.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)

    return hidden_state_dict, query_dict, key_dict, value_dict


def clean_attn_buffer(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and 'attn' in name:
            if 'injection_config' in module.processor.__dict__.keys():
                module.processor.injection_config = None
            if 'injection_mask' in module.processor.__dict__.keys():
                module.processor.injection_mask = None
            if 'obj_index' in module.processor.__dict__.keys():
                module.processor.obj_index = None
            if 'pca_weight' in module.processor.__dict__.keys():
                module.processor.pca_weight = None
            if 'pca_weight_changed' in module.processor.__dict__.keys():
                module.processor.pca_weight_changed = None
            if 'pca_info' in module.processor.__dict__.keys():
                module.processor.pca_info = None
            if 'step' in module.processor.__dict__.keys():
                module.processor.step = None
