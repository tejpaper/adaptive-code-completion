from pipeline.data.composed_datapoint import BatchComposedDatapoint

import math
import warnings
from abc import ABC, abstractmethod
from typing import TypedDict

import torch


class PreprocessedBatch(TypedDict):
    input_ids: torch.Tensor
    target_ids: torch.Tensor

    loss_mask: torch.Tensor
    completion_mask: torch.Tensor

    input_attn_mask: torch.Tensor
    target_attn_mask: torch.Tensor


class PreprocessorBase(ABC):
    tokenizer = None

    @abstractmethod
    def get_loss_mask(self, *args, **kwargs) -> torch.Tensor:
        """
        Important note: different number of masked tokens in different
        micro-batches will break gradient accumulation, in which case
        the training loop should include corresponding gradient scaling.
        *or we just don't care :)
        """
        raise NotImplementedError

    def calc_offset_mapping(self, seq_ids: list[int]) -> list[tuple[int, int]]:
        char_start = 0
        offsets = list()

        for token_len in map(len, self.tokenizer.batch_decode(seq_ids)):
            char_end = char_start + token_len
            offsets.append((char_start, char_end))
            char_start = char_end

        return offsets

    @abstractmethod
    def get_completion_mask(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, batch: BatchComposedDatapoint) -> PreprocessedBatch:
        raise NotImplementedError


class AmortizedPreprocessorBase(PreprocessorBase, ABC):
    def __init__(self, num_chars_per_token: int, verbose: bool) -> None:
        self.num_chars_per_token = num_chars_per_token
        self.verbose = verbose

    def _inc_num_chars_per_token(self) -> None:
        old_value = self.num_chars_per_token
        self.num_chars_per_token = math.ceil(1.5 * self.num_chars_per_token)

        if self.verbose:
            warnings.warn(
                f'num_chars_per_token has been increased from {old_value} to {self.num_chars_per_token} '
                'due to an underestimation of the length of the truncated character sequence.')
