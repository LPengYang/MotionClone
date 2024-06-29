import torch

def conv_forward(self):
    def forward(input_tensor, temb, scale=1.0):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)
        
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None].repeat(1, 1, hidden_states.shape[2], 1, 1)
            
        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # record hidden state
        self.record_hidden_state = hidden_states

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

    return forward


def get_conv_feat(unet):
    hidden_state_dict = dict()
    for i in range(len(unet.up_blocks)):
        for j in range(len(unet.up_blocks[i].resnets)):
            module = unet.up_blocks[i].resnets[j]
            module_name = f"up_blocks.{i}.resnets.{j}"
            # print(module_name)
            hidden_state_dict[module_name] = module.record_hidden_state
    return hidden_state_dict


def prep_unet_conv(unet):
    for i in range(len(unet.up_blocks)):
        for j in range(len(unet.up_blocks[i].resnets)):
            module = unet.up_blocks[i].resnets[j]
            module.forward = conv_forward(module)
    return unet
