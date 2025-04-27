from incontext.blocks.block import ComposerBlock
from incontext.data_structures import Datapoint

import random
from abc import ABC
from typing import Type


class ContextPostprocessor(ComposerBlock, ABC):
    last_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        return ContextPostprocessor,


class PartialMemoryPostprocessor(ContextPostprocessor):
    def __init__(self, dropout: float, random_seed: int | None) -> None:
        if not 0 <= dropout <= 1:
            raise ValueError('dropout must be selected from the interval [0, 1]. '
                             f'Got {dropout} instead.')
        self.dropout = dropout
        self.generator = random.Random(random_seed)

    def __call__(self, context: str, _datapoint: Datapoint) -> str:
        # dropping of path comments can occasionally happen
        return '\n'.join(line for line in context.split('\n') if self.generator.random() >= self.dropout)


class LineLengthPostprocessor(ContextPostprocessor):
    def __init__(self, min_len: int, max_len: int) -> None:
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, context: str, _datapoint: Datapoint) -> str:
        return '\n'.join(line for line in context.split('\n') if self.min_len <= len(line) <= self.max_len)


class LineStripPostprocessor(ContextPostprocessor):
    def __call__(self, context: str, _datapoint: Datapoint) -> str:
        return '\n'.join(line.strip() for line in context.split('\n'))
