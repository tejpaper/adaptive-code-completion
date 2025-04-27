from incontext.blocks.block import ComposerBlock
from incontext.data_structures import Chunk, Datapoint

from abc import ABC
from typing import Sequence, Type


class ChunkAssembler(ComposerBlock, ABC):
    last_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from incontext.blocks.context_postprocessing import ContextPostprocessor
        return ContextPostprocessor,


class JoiningAssembler(ChunkAssembler):
    def __init__(self, chunks_sep: str) -> None:
        self.chunks_sep = chunks_sep

    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> str:
        return self.chunks_sep.join(chunk.content for chunk in chunks)


class PathCommentAssembler(JoiningAssembler):
    def __init__(self, chunks_sep: str, path_comment_template: str) -> None:
        super().__init__(chunks_sep)
        self.path_comment_template = path_comment_template

    def __call__(self, chunks: Sequence[Chunk], datapoint: Datapoint) -> str:
        for chunk in chunks:
            chunk.content = self.path_comment_template.format(
                filename=chunk.file_ref.metadata['filename'], content=chunk.content)
        return super().__call__(chunks, datapoint)
