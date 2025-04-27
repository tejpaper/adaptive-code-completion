from incontext.data_structures import Chunk, Datapoint, File
from incontext.repr_mixin import ReprMixin

from abc import ABC, abstractmethod
from typing import Sequence

BlockArgs = Sequence[File] | Sequence[Chunk]


class ComposerBlock(ABC, ReprMixin):
    first_block_permit: bool = False
    last_block_permit: bool = False
    requires_tokenizer: bool = False

    @property
    @abstractmethod
    def next_blocks(self) -> tuple[type, ...]:
        raise NotImplementedError

    def check_next_block(self, block) -> None:
        if not isinstance(block, self.next_blocks):
            raise ValueError(f'{type(block).__name__} cannot be used after {type(self).__name__}.')

    @abstractmethod
    def __call__(self, args: BlockArgs, datapoint: Datapoint) -> BlockArgs | str:
        raise NotImplementedError
