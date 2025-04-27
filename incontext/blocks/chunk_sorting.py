from incontext.blocks.block import ComposerBlock
from incontext.data_structures import Chunk, Datapoint

from abc import ABC
from typing import Sequence, Type


class ChunkSorter(ComposerBlock, ABC):
    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from incontext.blocks.chunk_assembling import ChunkAssembler
        return ChunkSorter, ChunkAssembler


class LexicographicSorter(ChunkSorter):
    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        return sorted(chunks, key=lambda c: c.rank)


class ReverseLexicographicSorter(ChunkSorter):
    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        return sorted(chunks, key=lambda c: c.rank, reverse=True)


class MixedSorter(ChunkSorter):
    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        sorted_chunks = list()
        num_rankers = len(next(iter(chunks)).rank)
        i = 0

        for _ in range(len(chunks)):
            chunk = max(chunks, key=lambda x: x.rank[i])
            sorted_chunks.append(chunk)
            for j in range(num_rankers):
                chunk.rank[j] = -float('inf')
            i = (i + 1) % num_rankers

        return sorted_chunks[::-1]
