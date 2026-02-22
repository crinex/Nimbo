# Copyright (c) 2025, Nimbo
# Licensed under the Apache License, Version 2.0
# Based on Anemll (https://github.com/Anemll/Anemll) - MIT License

"""
ANE-optimized LLaMA model implementation for Apple Neural Engine.

This module provides LLaMA model components optimized for CoreML conversion
and Apple Neural Engine execution. Key optimizations include:

- Conv2d instead of Linear layers (better ANE utilization)
- Optimized RMSNorm using LayerNorm kernel
- Efficient KV cache management
- Float16 precision throughout

Supported models:
- LLaMA 3.2 (1B, 3B)
- LLaMA 3 (8B, 70B)
- LLaMA 2 (7B, 13B, 70B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import os
import json
import gc
from tqdm import tqdm
import safetensors
from abc import abstractmethod

# Model configuration constants
MODEL_DTYPE = torch.float16  # Hardcoded to float16 for ANE support
TEST_DEVICE = "cpu"  # Device for conversion (CPU required for tracing)

# Context and state length configuration
CONTEXT_LENGTH = 512  # Default context window size
STATE_LENGTH = 512    # KV cache state length

# Cache configuration
ENABLE_UNIFIED_CACHE = True  # Enable unified KV cache

# LM head configuration
ENABLE_CONV2D = True        # Use Conv2d for LM head
ENABLE_VOCAB_SPLIT = True   # Split vocab into 2 parts
ENABLE_VOCAB_SPLIT8 = True  # Split vocab into 8 parts

# MLP splits
MLP_UP_SPLIT = 1
MLP_DOWN_SPLIT = 1

# Activation functions
ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": F.silu,
}

# Debug flags (disable in production)
ENABLE_DEBUG = False
ENABLE_DEBUG2 = False
ENABLE_DEBUG3 = False
ENABLE_VALUES = False


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        pass


class LlamaConfig:
    """Configuration for ANE-optimized LLaMA model.

    This configuration class holds all the hyperparameters needed for
    the LLaMA model architecture.

    Attributes:
        hidden_size: Dimension of the hidden representations
        vocab_size: Size of the vocabulary
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        intermediate_size: Dimension of MLP intermediate layer
        hidden_act: Activation function name
        rms_norm_eps: Epsilon for RMSNorm
        rope_theta: Base for rotary position embeddings
        context_length: Maximum sequence length
        state_length: KV cache state length
    """

    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["LlamaForCausalLM"])
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.bos_token_id = kwargs.get("bos_token_id", 128000)
        self.eos_token_id = kwargs.get("eos_token_id", 128001)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.intermediate_size = kwargs.get("intermediate_size", 14336)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.model_type = kwargs.get("model_type", "llama")
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-05)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling.get("rope_type", "llama3")
        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.torch_dtype = kwargs.get("torch_dtype", "bfloat16")
        self.transformers_version = kwargs.get("transformers_version", "4.40.0")
        self.use_cache = kwargs.get("use_cache", True)
        self.vocab_size = kwargs.get("vocab_size", 128257)
        self.context_length = kwargs.get("context_length", CONTEXT_LENGTH)
        self.state_length = kwargs.get("state_length", STATE_LENGTH)

    @classmethod
    def from_json(cls, json_file):
        """Load configuration from JSON file."""
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in self.__dict__.items())


class LlamaRMSNorm(nn.Module):
    """ANE-optimized RMSNorm implementation.

    Uses LayerNorm kernel with a mean-subtraction bypass trick
    for better ANE performance.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compatibility path using LayerNorm kernel
        # Build tensor with mean = 0: concat([x, -x])
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)

        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * self.hidden_size,),
            weight=None,
            bias=None,
            eps=float(self.variance_epsilon)
        )

        # Drop the mirror half
        normed = normed[..., :self.hidden_size]

        # Apply learnable gain
        return (normed * self.weight.to(normed.dtype).to(normed.device))


class LlamaRotaryEmbedding(nn.Module):
    """Rotary positional embeddings for LLaMA model."""

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings

        # Apply rope_scaling factor if present
        self.base = config.rope_theta
        if hasattr(config, 'rope_scaling') and config.rope_scaling and 'factor' in config.rope_scaling:
            self.base = config.rope_theta * config.rope_scaling['factor']

        # Generate inverse frequency buffer
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(TEST_DEVICE) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache cos and sin values
        max_len = max(config.context_length, config.state_length) * 2
        t = torch.arange(max_len, device=TEST_DEVICE).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos().view(1, max_len, self.dim)
        self.sin_cached = emb.sin().view(1, max_len, self.dim)

    def forward(self, x, seq_len=None):
        return self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype)


class LlamaMLP(nn.Module):
    """ANE-optimized MLP using Conv2d layers."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Use Conv2d for ANE optimization
        self.gate_proj = nn.Conv2d(
            self.hidden_size, self.intermediate_size,
            kernel_size=1, bias=False, dtype=MODEL_DTYPE
        )
        self.up_proj = nn.Conv2d(
            self.hidden_size, self.intermediate_size,
            kernel_size=1, bias=False, dtype=MODEL_DTYPE
        )
        self.down_proj = nn.Conv2d(
            self.intermediate_size, self.hidden_size,
            kernel_size=1, bias=False, dtype=MODEL_DTYPE
        )

        self.act_fn = ACT2FN.get(config.hidden_act, F.silu)

    def forward(self, x):
        # Reshape for Conv2D: [bsz, seq_len, hidden] -> [bsz, hidden, 1, seq_len]
        x = x.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)

        gate_states = self.gate_proj(x)
        up_states = self.up_proj(x)

        gate_states = self.act_fn(gate_states)
        hidden_states = gate_states * up_states
        hidden_states = self.down_proj(hidden_states)

        # Reshape back: [bsz, hidden, 1, seq_len] -> [bsz, seq_len, hidden]
        return hidden_states.squeeze(2).permute(0, 2, 1)


class LlamaAttention(nn.Module):
    """ANE-optimized attention mechanism using Conv2d projections."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size: {self.hidden_size} and num_heads: {self.num_heads})"
            )

        # Use Conv2d for Q, K, V projections (ANE optimization)
        self.q_proj = nn.Conv2d(
            self.hidden_size, self.num_heads * self.head_dim,
            kernel_size=1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.k_proj = nn.Conv2d(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            kernel_size=1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        self.v_proj = nn.Conv2d(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            kernel_size=1, bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)
        # Output projection uses Linear
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=False, dtype=MODEL_DTYPE
        ).to(TEST_DEVICE)

        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.scaling_factor = torch.tensor(
            1.0 / math.sqrt(self.head_dim), dtype=MODEL_DTYPE, device=TEST_DEVICE
        )

    def get_new_kv_cache(self, hidden_states, current_pos, rotary_emb):
        """Get new key-value cache entries for single token generation."""
        # Project QKV
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        query_states = self.q_proj(hidden_states).view(
            1, self.num_heads, 1, self.head_dim
        ).to(MODEL_DTYPE)
        key_states = self.k_proj(hidden_states).view(
            1, self.num_key_value_heads, 1, self.head_dim
        ).to(MODEL_DTYPE)
        value_states = self.v_proj(hidden_states).view(
            1, self.num_key_value_heads, 1, self.head_dim
        ).to(MODEL_DTYPE)

        # Apply rotary embeddings
        cos, sin = rotary_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        return query_states, key_states, value_states

    def get_new_kv_cache_prefill(self, hidden_states, current_pos, rotary_emb, batch_size):
        """Get new key-value cache entries for prefill mode."""
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            1, self.num_heads, self.head_dim, -1
        ).permute(0, 1, 3, 2)
        key_states = key_states.view(
            1, self.num_key_value_heads, self.head_dim, -1
        ).permute(0, 1, 3, 2)
        value_states = value_states.view(
            1, self.num_key_value_heads, self.head_dim, -1
        ).permute(0, 1, 3, 2)

        cos, sin = rotary_emb
        cos = cos.permute(0, 2, 1, 3)
        sin = sin.permute(0, 2, 1, 3)

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        return query_states.to(MODEL_DTYPE), key_states.to(MODEL_DTYPE), value_states.to(MODEL_DTYPE)

    def apply_rotary_pos_emb(self, q_states, k_states, cos, sin):
        """Apply rotary position embeddings to query and key states."""

        def rotate(x, cos, sin):
            x = x.contiguous()
            half_dim = self.head_dim // 2

            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]

            if cos.dim() == 4:
                cos = cos[..., :half_dim]
                sin = sin[..., :half_dim]
            else:
                cos = cos.unsqueeze(1)[..., :half_dim]
                sin = sin.unsqueeze(1)[..., :half_dim]

            rotated = torch.cat([
                x1 * cos - x2 * sin,
                x2 * cos + x1 * sin
            ], dim=-1)

            return rotated.to(MODEL_DTYPE)

        return rotate(q_states, cos, sin), rotate(k_states, cos, sin)

    def ANE_softmax(self, x, dim=-1):
        """ANE-optimized softmax implementation."""
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x = x - x_max
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads for multi-head attention."""
        x = x.unsqueeze(1)
        x = x.repeat(1, n_rep, 1, 1)
        x = x.reshape(1, self.num_heads, -1, self.head_dim)
        return x

    def forward_regular(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None):
        """Forward pass for single token generation."""
        K_layer_cache, V_layer_cache = kv_cache_layer
        K_layer_cache = K_layer_cache[..., :self.config.context_length, :]
        V_layer_cache = V_layer_cache[..., :self.config.context_length, :]

        key_states = self.repeat_kv(K_layer_cache, self.num_key_value_groups)
        value_states = self.repeat_kv(V_layer_cache, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(-1, -2)
        ) * self.scaling_factor

        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask[:, :, :self.config.context_length]

        attn_weights = self.ANE_softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, -1, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def forward_prefill(self, hidden_states, query_states, kv_cache_layer=None, causal_mask=None):
        """Forward pass for prefill mode."""
        K_layer_cache, V_layer_cache = kv_cache_layer
        K_layer_cache = K_layer_cache[..., :self.config.context_length, :]
        V_layer_cache = V_layer_cache[..., :self.config.context_length, :]

        key_states = self.repeat_kv(K_layer_cache, self.num_key_value_groups)
        value_states = self.repeat_kv(V_layer_cache, self.num_key_value_groups)

        attn_weights = torch.einsum(
            'bhqd,bhkd->bhqk', query_states, key_states
        ) * self.scaling_factor

        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask[:, :, :self.config.context_length]

        attn_weights = self.ANE_softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, -1, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(nn.Module):
    """Transformer decoder layer for LLaMA."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def get_kv_cache_idx(layer_idx, num_layers, num_groups=1):
    """Helper function to get KV cache indices."""
    layers_per_group = num_layers // num_groups
    group_idx = layer_idx // layers_per_group
    layer_in_group_idx = layer_idx % layers_per_group
    return group_idx, layer_in_group_idx, layers_per_group


class LlamaModel(BaseModel):
    """ANE-optimized LLaMA transformer model."""

    def __init__(self, config, model_path=None):
        super().__init__(config)
        self.model_path = model_path

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Initialize unified KV cache
        cache_size = (
            2 * config.num_hidden_layers,
            config.num_key_value_heads,
            config.state_length,
            self.head_dim
        )
        self.register_buffer(
            "kv_cache_0",
            torch.zeros(cache_size, dtype=MODEL_DTYPE, device=TEST_DEVICE)
        )

    def get_rotary_embeddings_s(self, current_pos):
        """Get rotary embeddings for single position."""
        sin = self.layers[0].self_attn.rotary_emb.sin_cached[:, current_pos].view(1, 1, 1, -1)
        cos = self.layers[0].self_attn.rotary_emb.cos_cached[:, current_pos].view(1, 1, 1, -1)
        return cos, sin

    def get_rotary_embedding_prefill(self, positions):
        """Get rotary embeddings for multiple positions."""
        rotary_emb = self.layers[0].self_attn.rotary_emb
        cos = rotary_emb.cos_cached[:, positions].view(1, -1, 1, rotary_emb.dim)
        sin = rotary_emb.sin_cached[:, positions].view(1, -1, 1, rotary_emb.dim)
        return cos.to(MODEL_DTYPE), sin.to(MODEL_DTYPE)

    def process_layer(self, layer_idx, hidden_states, position_ids, causal_mask,
                      current_pos, rotary_emb, layer_offset, IN_PREFILL=False):
        """Process a single transformer layer."""
        layer = self.layers[layer_idx]
        batch_size = 1  # batch_size no longer used in get_new_kv_cache_prefill body

        normalized_states = layer.input_layernorm(hidden_states)

        # Get QKV states
        if IN_PREFILL:
            query_states, key_states, value_states = layer.self_attn.get_new_kv_cache_prefill(
                normalized_states, current_pos, rotary_emb, batch_size
            )
        else:
            query_states, key_states, value_states = layer.self_attn.get_new_kv_cache(
                normalized_states, current_pos, rotary_emb
            )

        # Get cache indices
        group_idx, layer_in_group_idx, layers_per_group = get_kv_cache_idx(
            layer_idx, self.config.num_hidden_layers
        )

        kv_cache = getattr(self, "kv_cache_0")
        key_idx = layer_in_group_idx
        value_idx = layer_in_group_idx + layers_per_group

        # Update KV cache
        if IN_PREFILL:
            seq_length = key_states.shape[2]
            kv_cache[key_idx:key_idx + 1, :, current_pos:current_pos + seq_length, :] = key_states
            kv_cache[value_idx:value_idx + 1, :, current_pos:current_pos + seq_length, :] = value_states
        else:
            kv_cache[key_idx:key_idx + 1, :, current_pos:current_pos + 1, :] = key_states
            kv_cache[value_idx:value_idx + 1, :, current_pos:current_pos + 1, :] = value_states

        key_cache = kv_cache[key_idx:key_idx + 1].squeeze(0)
        value_cache = kv_cache[value_idx:value_idx + 1].squeeze(0)

        # Run attention
        if IN_PREFILL:
            attn_output = layer.self_attn.forward_prefill(
                hidden_states=normalized_states,
                query_states=query_states,
                kv_cache_layer=(key_cache, value_cache),
                causal_mask=causal_mask,
            )
        else:
            attn_output = layer.self_attn.forward_regular(
                hidden_states=normalized_states,
                query_states=query_states,
                kv_cache_layer=(key_cache, value_cache),
                causal_mask=causal_mask,
            )

        hidden_states = hidden_states + attn_output

        # MLP (skip for last layer in prefill mode)
        is_last_layer = (layer_idx == len(self.layers) - 1)
        if not (IN_PREFILL and is_last_layer):
            post_attn = layer.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + layer.mlp(post_attn)

        return hidden_states

    def process_layers(self, hidden_states, position_ids, causal_mask,
                       current_pos, rotary_emb, start_layer=0, end_layer=None,
                       IN_PREFILL=False):
        """Process a range of transformer layers."""
        if end_layer is None:
            end_layer = len(self.layers)

        layer_offset = 0 if ENABLE_UNIFIED_CACHE else start_layer

        for i in range(start_layer, end_layer):
            hidden_states = self.process_layer(
                i, hidden_states, position_ids, causal_mask,
                current_pos, rotary_emb, layer_offset, IN_PREFILL
            )

        return hidden_states

    def forward(self, hidden_states, position_ids=None, causal_mask=None,
                current_pos=None, start_layer=0, end_layer=None, IN_PREFILL=False):
        """Forward pass with support for partial layer execution."""
        # Get rotary embeddings
        if IN_PREFILL:
            rotary_emb = self.get_rotary_embedding_prefill(position_ids)
        else:
            rotary_emb = self.get_rotary_embeddings_s(current_pos)

        # Process layers
        hidden_states = self.process_layers(
            hidden_states, position_ids, causal_mask,
            current_pos, rotary_emb, start_layer, end_layer, IN_PREFILL=IN_PREFILL
        )

        # Apply final normalization if last block
        if end_layer is None or end_layer == len(self.layers):
            if IN_PREFILL:
                return hidden_states[:, 0:1, :]
            else:
                hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(nn.Module):
    """LLaMA model with causal language modeling head.

    This is the main model class for text generation, combining
    the transformer backbone with an LM head for next token prediction.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, enable_coreml=False):
        super().__init__()
        self.config = config
        self.enable_coreml = enable_coreml
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(TEST_DEVICE)
        self.model = LlamaModel(config).to(TEST_DEVICE)

        # Initialize LM head as Conv2d for ANE optimization
        if ENABLE_CONV2D:
            if ENABLE_VOCAB_SPLIT8:
                for i in range(1, 9):
                    setattr(self, f'lm_head8_{i}', nn.Conv2d(
                        config.hidden_size, config.vocab_size // 8,
                        1, bias=False, dtype=MODEL_DTYPE
                    ).to(TEST_DEVICE))
            elif ENABLE_VOCAB_SPLIT:
                self.lm_head2_1 = nn.Conv2d(
                    config.hidden_size, config.vocab_size // 2,
                    1, bias=False, dtype=MODEL_DTYPE
                ).to(TEST_DEVICE)
                self.lm_head2_2 = nn.Conv2d(
                    config.hidden_size, config.vocab_size // 2,
                    1, bias=False, dtype=MODEL_DTYPE
                ).to(TEST_DEVICE)
            else:
                self.lm_head1 = nn.Conv2d(
                    config.hidden_size, config.vocab_size,
                    1, bias=False, dtype=MODEL_DTYPE
                ).to(TEST_DEVICE)
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size,
                bias=False, dtype=MODEL_DTYPE
            ).to(TEST_DEVICE)

    def load_pretrained_weights(self, model_path, **kwargs):
        """Load pretrained weights from safetensors files."""
        print("Loading pretrained weights...")

        file_dict = {}
        for file in tqdm(os.listdir(model_path)):
            if file.endswith(".safetensors"):
                file_dict.update(safetensors.torch.load_file(os.path.join(model_path, file)))

        # Handle lm_head weight
        lm_head_present = "lm_head.weight" in file_dict
        embed_tokens_key = None
        for k in file_dict.keys():
            if "embed_tokens.weight" in k:
                embed_tokens_key = k
                break

        if not lm_head_present and embed_tokens_key:
            file_dict['lm_head.weight'] = file_dict[embed_tokens_key].clone()

        # Filter and reshape weights
        filtered_state_dict = {}
        for k, v in file_dict.items():
            if k == "model.embed_tokens.weight":
                filtered_state_dict["embed_tokens.weight"] = v
            elif k == "lm_head.weight":
                if ENABLE_CONV2D:
                    reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                    if ENABLE_VOCAB_SPLIT8:
                        vocab_split = self.config.vocab_size // 8
                        splits = torch.split(reshaped_weight, vocab_split)
                        for i, split in enumerate(splits):
                            filtered_state_dict[f"lm_head8_{i+1}.weight"] = split
                    elif ENABLE_VOCAB_SPLIT:
                        vocab_split = self.config.vocab_size // 2
                        split1, split2 = torch.split(
                            reshaped_weight,
                            [vocab_split, self.config.vocab_size - vocab_split]
                        )
                        filtered_state_dict["lm_head2_1.weight"] = split1
                        filtered_state_dict["lm_head2_2.weight"] = split2
                    else:
                        filtered_state_dict["lm_head1.weight"] = reshaped_weight
                else:
                    filtered_state_dict["lm_head.weight"] = v

        self.load_state_dict(filtered_state_dict, strict=False)

        # Load base model weights
        base_filtered_dict = {}
        for k, v in file_dict.items():
            if k.startswith("model."):
                new_key = k.replace("model.", "")
                if "layers." in new_key:
                    if 'self_attn' in new_key and 'weight' in new_key:
                        if 'o_proj' in new_key:
                            base_filtered_dict[new_key] = v
                        else:
                            reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                            base_filtered_dict[new_key] = reshaped_weight
                    elif 'mlp' in new_key and 'weight' in new_key:
                        reshaped_weight = v.view(v.shape[0], v.shape[1], 1, 1)
                        base_filtered_dict[new_key] = reshaped_weight
                    else:
                        base_filtered_dict[new_key] = v
                elif new_key == "norm.weight":
                    base_filtered_dict[new_key] = v

        self.model.load_state_dict(base_filtered_dict, strict=False)
        print("Pretrained weights loaded successfully")
        return True

    def forward(
        self,
        input_ids: torch.LongTensor,
        update_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        current_pos: int,
        causal_mask: torch.Tensor,
        IN_PREFILL: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for causal language modeling."""
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.to(MODEL_DTYPE)

        hidden_states = self.model(
            hidden_states=hidden_states,
            position_ids=position_ids,
            current_pos=current_pos,
            causal_mask=causal_mask,
            start_layer=0,
            end_layer=None,
            IN_PREFILL=IN_PREFILL,
        )

        # Project to vocabulary
        if ENABLE_CONV2D:
            hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

            if ENABLE_VOCAB_SPLIT8:
                logits_list = [
                    getattr(self, f'lm_head8_{i}')(hidden_states).squeeze(2).transpose(1, 2)
                    for i in range(1, 9)
                ]
                logits = torch.cat(logits_list, dim=2)
            elif ENABLE_VOCAB_SPLIT:
                logits1 = self.lm_head2_1(hidden_states).squeeze(2).transpose(1, 2)
                logits2 = self.lm_head2_2(hidden_states).squeeze(2).transpose(1, 2)
                logits = torch.cat([logits1, logits2], dim=2)
            else:
                logits = self.lm_head1(hidden_states).squeeze(2).transpose(1, 2)
        else:
            logits = self.lm_head(hidden_states)

        return logits
