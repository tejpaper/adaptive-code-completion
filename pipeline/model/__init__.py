from pipeline.environment.hardware import get_free_device, get_optimal_dtype

from enum import Enum
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available


class AttentionImplementation(str, Enum):
    FA2 = 'flash_attention_2'
    SDPA = 'sdpa'
    EAGER = 'eager'


def init_tokenizer(tokenizer_name: str, trust_remote_code: bool, **_kwargs) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name,
        trust_remote_code=trust_remote_code,
    )

    tokenizer.padding_side = 'right'
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<|SEP|>'})

    return tokenizer


def get_optimal_attn(model_name: str, device: torch.device, dtype: torch.dtype) -> AttentionImplementation:
    hf_model_config = AutoConfig.from_pretrained(model_name)
    model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(hf_model_config)]

    fa2_supported = (
            is_flash_attn_2_available() and
            model_class._supports_flash_attn_2 and  # noqa: HF doesn't have an API for this case
            device.type == 'cuda' and
            dtype in (torch.float16, torch.bfloat16)
    )

    if fa2_supported:
        return AttentionImplementation.FA2
    elif is_torch_sdpa_available() and model_class._supports_sdpa:  # noqa: same
        return AttentionImplementation.SDPA
    else:
        return AttentionImplementation.EAGER


def init_model(model_name: str,
               trust_remote_code: bool,
               use_cache: bool = False,
               device: str | None = None,
               dtype: str | None = None,
               attn_implementation: AttentionImplementation | None = None,
               compile: bool = True,
               config: dict[str, Any] | None = None,
               **_kwargs,
               ) -> PreTrainedModel:
    device = get_free_device() if device is None else torch.device(device)
    dtype = get_optimal_dtype() if dtype is None else getattr(torch, dtype)

    if attn_implementation is None:
        attn_implementation = get_optimal_attn(model_name, device, dtype)
    else:
        attn_implementation = AttentionImplementation(attn_implementation)

    if config is None:
        config = dict()

    kwargs_config = config
    if not isinstance(kwargs_config, dict):
        kwargs_config = OmegaConf.to_container(kwargs_config)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=trust_remote_code,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        use_cache=use_cache,
        **kwargs_config,
    )

    if compile:
        return torch.compile(model)
    else:
        return model


__all__ = [
    'init_tokenizer',
    'init_model',
]