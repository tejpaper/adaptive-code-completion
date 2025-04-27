from incontext.blocks.block import ComposerBlock
from incontext.composer.composer_base import ComposerBase
from incontext.data_structures import Datapoint, File
from incontext.repr_mixin import ReprMixin

from typing import Any, Sequence


class UnsafeComposerChain:
    def __init__(self, *blocks: ComposerBlock) -> None:
        self.blocks = blocks

    def __call__(self, datapoint: Datapoint) -> Any:
        x = [
            File(content=cnt, metadata={'filename': fn})
            for fn, cnt in zip(*datapoint.repo_snapshot.values())
        ]
        for block in self.blocks:
            x = block(x, datapoint)
        return x


class ComposerChain(UnsafeComposerChain):
    def __init__(self, *blocks: ComposerBlock) -> None:
        if not blocks:
            raise ValueError('ComposerChain instance must contain at least one element.')
        elif not blocks[0].first_block_permit:
            raise ValueError(f'{type(blocks[0]).__name__} cannot start a chain of blocks.')
        elif not blocks[-1].last_block_permit:
            raise ValueError(f'{type(blocks[-1]).__name__} cannot end a chain of blocks.')

        for block, next_block in zip(blocks[:-1], blocks[1:]):
            block.check_next_block(next_block)

        super().__init__(*blocks)


class ChainedComposer(ComposerBase, ComposerChain, ReprMixin):
    def __init__(self, blocks: Sequence[ComposerBlock], *args, **kwargs) -> None:
        ComposerBase.__init__(self, *args, **kwargs)
        ComposerChain.__init__(self, *blocks)

    def compose_context(self, datapoint: Datapoint) -> str:
        return self.__call__(datapoint)
