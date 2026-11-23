from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.lora import LoRALinearLayer
from einops import rearrange, repeat
from diffusers.models.lora import LoRALinearLayer



class CacheAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.cache = {}  # cache hidden states

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
    ) -> torch.FloatTensor:

        self.cache["hidden_states"] = hidden_states  # cache hidden states

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            # for reference adapter
            if sa_hidden_states is not None:
                ref_hidden_states = sa_hidden_states[self.name]
                encoder_hidden_states = torch.cat([hidden_states, ref_hidden_states], dim=1)
            else:
                encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class CAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim



    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
class SKIPAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim



    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
    ) -> torch.FloatTensor:
        return hidden_states


class BaseSAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.cross_attention_dim = cross_attention_dim

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



class RefSAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.to_k_ref = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ref = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.scale = scale

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            num_images_per_prompt=1,
            cond_hidden_states=None,
            sa_hidden_states=None,

    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ref adapter
        if sa_hidden_states is not None:
            ref_hidden_states = sa_hidden_states[self.name]
            # for ref
            ref_key = self.to_k_ref(ref_hidden_states)
            ref_value = self.to_v_ref(ref_hidden_states)
            ref_key = ref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_value = ref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ref_hidden_states = F.scaled_dot_product_attention(
                query, ref_key, ref_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            ref_hidden_states = ref_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ref_hidden_states = ref_hidden_states.to(query.dtype)
            hidden_states = hidden_states + ref_hidden_states * self.scale

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class RefCAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_k_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v_ref = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = scale

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ref adapter
        if sa_hidden_states is not None:
            ref_hidden_states = sa_hidden_states[self.name]
            # for ref
            ref_key = self.to_k_ref(ref_hidden_states)
            ref_value = self.to_v_ref(ref_hidden_states)
            ref_key = ref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_value = ref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ref_hidden_states = F.scaled_dot_product_attention(
                query, ref_key, ref_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            ref_hidden_states = ref_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ref_hidden_states = ref_hidden_states.to(query.dtype)
            hidden_states = hidden_states + ref_hidden_states * self.scale

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states





class LogoRefSAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None, scale=1.0 , logo_scale=1.0 , color_scale=1.0,
                 skip_sketch = False, skip_logo = False, skip_color = False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        if not skip_sketch:
            self.to_k_ref = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_ref = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        if not skip_logo:
            self.to_k_ref_logo = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_ref_logo = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        if not skip_color:
            self.to_k_ref_color = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_ref_color = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.scale = scale
        self.logo_scale = logo_scale
        self.color_scale = color_scale

        self.skip_sketch = skip_sketch
        self.skip_logo = skip_logo
        self.skip_color = skip_color

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            num_images_per_prompt=1,
            cond_hidden_states=None,
            sa_hidden_states=None,

    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # for ref adapter
        if sa_hidden_states is not None:
            
            ref_hidden_states = sa_hidden_states[self.name]
            ref_hidden_states,ref_logo_hidden_states,ref_color_hidden_states = torch.chunk(ref_hidden_states,3,dim=0)
            
            # for ref sketch
            if not self.skip_sketch:
                ref_key = self.to_k_ref(ref_hidden_states)
                ref_value = self.to_v_ref(ref_hidden_states)
                ref_key = ref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ref_value = ref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                ref_hidden_states = F.scaled_dot_product_attention(
                    query, ref_key, ref_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                ref_hidden_states = ref_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ref_hidden_states = ref_hidden_states.to(query.dtype)
                
            
            # for ref logo
            if not self.skip_logo:
                ref_logo_key = self.to_k_ref_logo(ref_logo_hidden_states)
                ref_logo_value = self.to_v_ref_logo(ref_logo_hidden_states)
                ref_logo_key = ref_logo_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ref_logo_value = ref_logo_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                ref_logo_hidden_states = F.scaled_dot_product_attention(
                    query, ref_logo_key, ref_logo_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                ref_logo_hidden_states = ref_logo_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ref_logo_hidden_states = ref_logo_hidden_states.to(query.dtype)
            
            # for ref color
            if not self.skip_color:
                ref_color_key = self.to_k_ref_color(ref_color_hidden_states)
                ref_color_value = self.to_v_ref_color(ref_color_hidden_states)
                ref_color_key = ref_color_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ref_color_value = ref_color_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                ref_color_hidden_states = F.scaled_dot_product_attention(
                    query, ref_color_key, ref_color_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                ref_color_hidden_states = ref_color_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ref_color_hidden_states = ref_color_hidden_states.to(query.dtype)
            
            if not self.skip_sketch:
                hidden_states = hidden_states + ref_hidden_states * self.scale
            if not self.skip_logo:
                hidden_states = hidden_states + ref_logo_hidden_states * self.logo_scale
            if not self.skip_color:
                hidden_states = hidden_states + ref_color_hidden_states * self.color_scale
            # hidden_states = hidden_states + ref_hidden_states * self.scale + ref_logo_hidden_states * self.logo_scale + ref_color_hidden_states * self.color_scale

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LogoCacheSAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None,scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.cache = {}  # cache hidden states
        
        self.to_q_logo = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_logo = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_logo = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.to_q_color = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_color = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_color = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        
    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
    ) -> torch.FloatTensor:
        self.cache["hidden_states"] = hidden_states  # cache hidden states
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        batch_size=batch_size // 3
        
        hidden_states,logo_hidden_states,color_hidden_states=torch.chunk(hidden_states, 3, dim=0)

        
        #sketch
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        #logo
        logo_query = self.to_q_logo(logo_hidden_states)

        logo_key = self.to_k_logo(logo_hidden_states)
        logo_value =self.to_v_logo(logo_hidden_states)

        inner_dim = logo_key.shape[-1]
        head_dim = inner_dim // attn.heads

        logo_query = logo_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        logo_key = logo_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        logo_value = logo_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        logo_hidden_states = F.scaled_dot_product_attention(
            logo_query, logo_key, logo_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        logo_hidden_states = logo_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        logo_hidden_states = logo_hidden_states.to(query.dtype)
        
        
        #mask
        color_query = self.to_q_color(color_hidden_states)

        color_key = self.to_k_color(color_hidden_states)
        color_value =self.to_v_color(color_hidden_states)

        inner_dim = color_key.shape[-1]
        head_dim = inner_dim // attn.heads

        color_query = color_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        color_key = color_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        color_value = color_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        color_hidden_states = F.scaled_dot_product_attention(
            color_query, color_key, color_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        color_hidden_states = color_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        color_hidden_states = color_hidden_states.to(query.dtype)
        
        
        hidden_states=torch.cat([hidden_states,logo_hidden_states,color_hidden_states],dim=0)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states

class LogoCacheCAttnProcessor2_0(torch.nn.Module):
    
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.to_q_logo = nn.Linear(  hidden_size, hidden_size, bias=False)
        self.to_k_logo = nn.Linear( cross_attention_dim, hidden_size, bias=False)
        self.to_v_logo = nn.Linear( cross_attention_dim, hidden_size, bias=False)

        self.to_q_color = nn.Linear(  hidden_size, hidden_size, bias=False)
        self.to_k_color = nn.Linear( cross_attention_dim, hidden_size, bias=False)
        self.to_v_color = nn.Linear( cross_attention_dim, hidden_size, bias=False)
    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # batch_size = batch_size//3

        hidden_states,logo_hidden_states,color_hidden_states=torch.chunk(hidden_states, 3, dim=0)

        # sketch 
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        # print(hidden_states.shape)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        # print(query.shape)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        #logo
        logo_query = self.to_q_logo(logo_hidden_states)

        logo_key = self.to_k_logo(encoder_hidden_states)
        logo_value =self.to_v_logo(encoder_hidden_states)

        inner_dim = logo_key.shape[-1]
        head_dim = inner_dim // attn.heads

        logo_query = logo_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        logo_key = logo_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        logo_value = logo_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        logo_hidden_states = F.scaled_dot_product_attention(
            logo_query, logo_key, logo_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        logo_hidden_states = logo_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        logo_hidden_states = logo_hidden_states.to(hidden_states.dtype)

        # color
        color_query = self.to_q_color(color_hidden_states)

        color_key = self.to_k_color(encoder_hidden_states)
        color_value =self.to_v_color(encoder_hidden_states)

        inner_dim = color_key.shape[-1]
        head_dim = inner_dim // attn.heads

        color_query = color_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        color_key = color_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        color_value = color_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        color_hidden_states = F.scaled_dot_product_attention(
            color_query, color_key, color_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        color_hidden_states = color_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        color_hidden_states = color_hidden_states.to(query.dtype)

        hidden_states=torch.cat([hidden_states,logo_hidden_states,color_hidden_states],dim=0)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
class LoraLogoCacheSAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None,rank=128,network_alpha=None,lora_scale=1.0,skip_sketch = False,skip_color = False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.cache = {}  # cache hidden states
        self.skip_sketch = skip_sketch
        self.skip_color = skip_color
        self.lora_scale = lora_scale
        self.rank = rank
        

        self.to_logo_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_logo_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_logo_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

        if not skip_sketch:
            self.to_sketch_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
            self.to_sketch_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
            self.to_sketch_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
            
        if not skip_color:
            self.to_color_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
            self.to_color_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
            self.to_color_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        
        
    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
    ) -> torch.FloatTensor:
        self.cache["hidden_states"] = hidden_states  # cache hidden states
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        batch_size=batch_size // 3
        
        hidden_states,logo_hidden_states,color_hidden_states=torch.chunk(hidden_states, 3, dim=0)

        
        #sketch
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        if not self.skip_sketch:
            query = attn.to_q(hidden_states) + self.lora_scale * self.to_sketch_k_lora(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


            key = attn.to_k(hidden_states) + self.lora_scale * self.to_sketch_q_lora(hidden_states)
            value = attn.to_v(hidden_states) + self.lora_scale * self.to_sketch_v_lora(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

        #logo
        logo_query = attn.to_q(logo_hidden_states) + self.lora_scale * self.to_logo_q_lora(logo_hidden_states)

        logo_key = attn.to_k(logo_hidden_states) + self.lora_scale * self.to_logo_k_lora(logo_hidden_states)
        logo_value = attn.to_v(logo_hidden_states) + self.lora_scale * self.to_logo_v_lora(logo_hidden_states)

        inner_dim = logo_key.shape[-1]
        head_dim = inner_dim // attn.heads

        logo_query = logo_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        logo_key = logo_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        logo_value = logo_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        logo_hidden_states = F.scaled_dot_product_attention(
            logo_query, logo_key, logo_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        logo_hidden_states = logo_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        logo_hidden_states = logo_hidden_states.to(logo_query.dtype)
        
        
        # color
        if not self.skip_color:
            color_query = attn.to_q(color_hidden_states) + self.lora_scale * self.to_color_q_lora(color_hidden_states)

            color_key = attn.to_k(color_hidden_states) + self.lora_scale * self.to_color_k_lora(color_hidden_states)
            color_value = attn.to_v(color_hidden_states) + self.lora_scale * self.to_color_v_lora(color_hidden_states)

            inner_dim = color_key.shape[-1]
            head_dim = inner_dim // attn.heads

            color_query = color_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            color_key = color_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            color_value = color_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            color_hidden_states = F.scaled_dot_product_attention(
                color_query, color_key, color_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            color_hidden_states = color_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            color_hidden_states = color_hidden_states.to(color_query.dtype)
            
        
        hidden_states=torch.cat([hidden_states,logo_hidden_states,color_hidden_states],dim=0)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states

class LoraLogoCacheCAttnProcessor2_0(torch.nn.Module):
    
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None,rank=128, network_alpha=None,
                 lora_scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_q_logo = nn.Linear(  hidden_size, hidden_size, bias=False)
        self.to_k_logo = nn.Linear( cross_attention_dim, hidden_size, bias=False)
        self.to_v_logo = nn.Linear( cross_attention_dim, hidden_size, bias=False)

        self.to_q_color = nn.Linear(  hidden_size, hidden_size, bias=False)
        self.to_k_color = nn.Linear( cross_attention_dim, hidden_size, bias=False)
        self.to_v_color = nn.Linear( cross_attention_dim, hidden_size, bias=False)


        self.rank = rank
        self.lora_scale = lora_scale
        self.to_q_sketch_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_sketch_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_sketch_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

        self.to_q_logo_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_logo_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_logo_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

        self.to_q_color_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_color_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_color_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        
    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            cond_hidden_states=None,
            sa_hidden_states=None,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # batch_size = batch_size//3

        hidden_states,logo_hidden_states,color_hidden_states=torch.chunk(hidden_states, 3, dim=0)

        # sketch 
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        # print(hidden_states.shape)
        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_sketch_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_sketch_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_sketch_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        # print(query.shape)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        #logo
        logo_query = self.to_q_logo(logo_hidden_states) + self.lora_scale * self.to_q_logo_lora(logo_hidden_states)

        logo_key = self.to_k_logo(encoder_hidden_states) + self.lora_scale * self.to_k_logo_lora(encoder_hidden_states)
        logo_value = self.to_v_logo(encoder_hidden_states) + self.lora_scale * self.to_v_logo_lora(encoder_hidden_states)

        inner_dim = logo_key.shape[-1]
        head_dim = inner_dim // attn.heads

        logo_query = logo_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        logo_key = logo_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        logo_value = logo_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        logo_hidden_states = F.scaled_dot_product_attention(
            logo_query, logo_key, logo_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        logo_hidden_states = logo_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        logo_hidden_states = logo_hidden_states.to(query.dtype)

        # color 
        color_query = self.to_q_color(color_hidden_states) + self.lora_scale * self.to_q_color_lora(color_hidden_states)

        color_key = self.to_k_color(encoder_hidden_states) + self.lora_scale * self.to_k_color_lora(encoder_hidden_states)
        color_value = self.to_v_color(encoder_hidden_states) + self.lora_scale * self.to_v_color_lora(encoder_hidden_states)

        inner_dim = color_key.shape[-1]
        head_dim = inner_dim // attn.heads

        color_query = color_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        color_key = color_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        color_value = color_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        color_hidden_states = F.scaled_dot_product_attention(
            color_query, color_key, color_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        color_hidden_states = color_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        color_hidden_states = color_hidden_states.to(query.dtype)

        hidden_states=torch.cat([hidden_states,logo_hidden_states,color_hidden_states],dim=0)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states) +  self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
class LoraLogoRefSAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, hidden_size, cross_attention_dim=None, scale=1.0 , logo_scale=1.0 , color_scale=1.0,rank=128, network_alpha=None,
                 lora_scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        super().__init__()

        self.name = name
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.rank = rank
        self.lora_scale = lora_scale

        self.to_k_ref = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ref = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.to_k_ref_logo = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ref_logo = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.to_k_ref_color = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ref_color = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)


        self.to_k_sketch_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_sketch_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

        self.to_k_logo_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_logo_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

        self.to_k_color_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_color_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        self.scale = scale
        self.logo_scale = logo_scale
        self.color_scale = color_scale

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            num_images_per_prompt=1,
            cond_hidden_states=None,
            sa_hidden_states=None,

    ) -> torch.FloatTensor:
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # for ref adapter
        if sa_hidden_states is not None:
            ref_hidden_states = sa_hidden_states[self.name]
            ref_hidden_states,ref_logo_hidden_states,ref_color_hidden_states = torch.chunk(ref_hidden_states,3,dim=0)
            
            # for ref sketch
            ref_key = self.to_k_ref(ref_hidden_states) +  self.lora_scale * self.to_k_sketch_lora(ref_hidden_states)
            ref_value = self.to_v_ref(ref_hidden_states) + self.lora_scale * self.to_v_sketch_lora(ref_hidden_states)
            ref_key = ref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_value = ref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ref_hidden_states = F.scaled_dot_product_attention(
                query, ref_key, ref_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            ref_hidden_states = ref_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ref_hidden_states = ref_hidden_states.to(query.dtype)
            
            
            # for ref logo
            ref_logo_key = self.to_k_ref_logo(ref_logo_hidden_states) + self.lora_scale * self.to_k_logo_lora(ref_logo_hidden_states)
            ref_logo_value = self.to_v_ref_logo(ref_logo_hidden_states) + self.lora_scale * self.to_v_logo_lora(ref_logo_hidden_states)
            ref_logo_key = ref_logo_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_logo_value = ref_logo_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ref_logo_hidden_states = F.scaled_dot_product_attention(
                query, ref_logo_key, ref_logo_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            ref_logo_hidden_states = ref_logo_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ref_logo_hidden_states = ref_logo_hidden_states.to(query.dtype)
            
            # for ref color
            ref_color_key = self.to_k_ref_color(ref_color_hidden_states) + self.lora_scale * self.to_k_color_lora(ref_color_hidden_states)
            ref_color_value = self.to_v_ref_color(ref_color_hidden_states) + self.color_scale * self.to_v_color_lora(ref_color_hidden_states)
            ref_color_key = ref_color_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_color_value = ref_color_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ref_color_hidden_states = F.scaled_dot_product_attention(
                query, ref_color_key, ref_color_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            ref_color_hidden_states = ref_color_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ref_color_hidden_states = ref_color_hidden_states.to(query.dtype)
            
            
            hidden_states = hidden_states + ref_hidden_states * self.scale + ref_logo_hidden_states * self.logo_scale + ref_color_hidden_states * self.color_scale

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states