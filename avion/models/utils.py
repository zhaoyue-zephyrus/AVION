from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_grad_checkpointing(model: nn.Module, enable: bool):
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(enable=enable)
    else:
        print("{} has no attribute named 'set_grad_checkpointing'".format(model._get_name()))


# util functions to convert OpenCLIP-style model keys to ViT-style
def remap_keys_from_open_clip_to_vit(
    clip_state_dict,
    visual_transformer_layers=12,
    textual_transformer_layers=12,
    context_length=77,
    vocab_size=49408,
    use_fast_conv1=False,
    use_flash_attn=False,
):
    if 'state_dict' in clip_state_dict:
        clip_state_dict = clip_state_dict['state_dict']
    if list(clip_state_dict.keys())[0].startswith('module.'):
        clip_state_dict = OrderedDict({
            k.replace('module.', ''): v for k, v in clip_state_dict.items()
        })
    remapped_state_dict = OrderedDict()
    key_mapping = {
        "logit_scale": "logit_scale",
        "visual.proj": "visual.image_projection",
        "positional_embedding": "textual.positional_embedding",
        "text_projection": "textual.text_projection",
        "token_embedding.weight": "textual.token_embedding.weight",
        "ln_final.weight": "textual.ln_final.weight",
        "ln_final.bias": "textual.ln_final.bias"
    }

    for layer in range(visual_transformer_layers):
        if use_flash_attn:
            for src_name, tgt_name in {
                'attn.in_proj_weight': 'attn.Wqkv.weight', 'attn.in_proj_bias': 'attn.Wqkv.bias',
                'attn.out_proj.weight': 'attn.out_proj.weight', 'attn.out_proj.bias': 'attn.out_proj.bias',
                'mlp.c_fc.weight': 'mlp.fc1.weight', 'mlp.c_fc.bias': 'mlp.fc1.bias',
                'mlp.c_proj.weight': 'mlp.fc2.weight', 'mlp.c_proj.bias': 'mlp.fc2.bias',
            }.items():
                key_mapping[f"visual.transformer.resblocks.{layer}.{src_name}"] = f"visual.transformer.resblocks.{layer}.{tgt_name}"


    for layer in range(textual_transformer_layers):
        for name in [
            'attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias',
            'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias',
             'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias',
        ]:
            key_mapping[f"transformer.resblocks.{layer}.{name}"] = f"textual.transformer.resblocks.{layer}.{name}"

    for key in clip_state_dict:
        if key in ["visual.proj", "text_projection", "logit_scale"]:
            continue
        if use_fast_conv1 and key == 'visual.conv1.weight':
            remapped_state_dict['visual.conv1.weight'] = clip_state_dict[key].flatten(1)
            # assert mean is not None and std is not None
            # W_2 = clip_state_dict[key].flatten(1)
            # std = torch.tensor(std).float()
            # std = std.repeat_interleave(clip_state_dict[key].shape[-1] * clip_state_dict[key].shape[-2])
            # W_1 = torch.diag(1 / std)
            # W_fused = W_2 @ W_1
            # mean = torch.tensor(mean).float().repeat_interleave(clip_state_dict[key].shape[-1] * clip_state_dict[key].shape[-2])
            # b_1 = mean / std
            # b_fused = W_2 @ (-b_1)
            # remapped_state_dict['visual.conv1.weight'] = W_fused
            # remapped_state_dict['visual.conv1.bias'] = b_fused
        elif key not in key_mapping:
            remapped_state_dict[key] = clip_state_dict[key]
        else:
            if key == 'positional_embedding':
                old_context_length, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                if context_length <= old_context_length:
                    remapped_state_dict[key_mapping[key]] = clip_state_dict[key][:context_length, :]
                else:
                    remapped_state_dict[key_mapping[key]] = torch.cat(
                        (clip_state_dict[key], torch.zeros((context_length - old_context_length, dim), dtype=old_dtype)), dim=0
                    )
            elif key == 'token_embedding.weight':
                old_vocab_size, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                assert vocab_size >= old_vocab_size
                remapped_state_dict[key_mapping[key]] = torch.cat(
                    (clip_state_dict[key], torch.zeros((vocab_size - old_vocab_size, dim), dtype=old_dtype)), dim=0
                )
            else:
                remapped_state_dict[key_mapping[key]] = clip_state_dict[key]

    return remapped_state_dict


def inflate_positional_embeds(
    current_model_state_dict, new_state_dict,
    num_frames=4,
    load_temporal_fix='bilinear',
):
    # allow loading of timesformer with fewer num_frames
    curr_keys = list(current_model_state_dict.keys())
    if 'visual.temporal_embedding' in new_state_dict and 'visual.temporal_embedding' in curr_keys:
        load_temporal_embed = new_state_dict['visual.temporal_embedding']
        load_num_frames = load_temporal_embed.shape[0]
        curr_num_frames = num_frames
        embed_dim = load_temporal_embed.shape[1]

        if load_num_frames != curr_num_frames:
            if load_num_frames > curr_num_frames:
                print(f'### loaded SpaceTimeTransformer model has MORE frames than current...'
                      f'### loading weights, filling in the extras via {load_temporal_fix}')
                new_temporal_embed = load_temporal_embed[:curr_num_frames, :]
            else:
                print(f'### loaded SpaceTimeTransformer model has FEWER frames than current...'
                      f'### loading weights, filling in the extras via {load_temporal_fix}')
                if load_temporal_fix == 'zeros':
                    new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                    new_temporal_embed[:load_num_frames] = load_temporal_embed
                elif load_temporal_fix in ['interp', 'bilinear']:
                    # interpolate
                    # unsqueeze so pytorch thinks its an image
                    mode = 'nearest'
                    if load_temporal_fix == 'bilinear':
                        mode = 'bilinear'
                    load_temporal_embed = load_temporal_embed.unsqueeze(0).unsqueeze(0)
                    new_temporal_embed = F.interpolate(load_temporal_embed,
                                                       (curr_num_frames, embed_dim), mode=mode).squeeze(0).squeeze(0)
                else:
                    raise NotImplementedError
            new_state_dict['visual.temporal_embedding'] = new_temporal_embed
    # allow loading with smaller spatial patches. assumes custom border crop, to append the
    # border patches to the input sequence
    if 'visual.positional_embedding' in new_state_dict and 'visual.positional_embedding' in curr_keys:
        load_pos_embed = new_state_dict['visual.positional_embedding']
        load_num_patches = load_pos_embed.shape[0]
        curr_pos_embed = current_model_state_dict['visual.positional_embedding']
        if load_num_patches != curr_pos_embed.shape[0]:
            raise NotImplementedError(
                'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

    return new_state_dict
