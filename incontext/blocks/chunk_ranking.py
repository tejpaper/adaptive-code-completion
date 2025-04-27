from incontext.blocks.block import ComposerBlock
from incontext.data_structures import Chunk, Datapoint

import os
import random
import warnings
from abc import ABC
from typing import Sequence, Type

import tree_sitter
import tree_sitter_python


class ChunkRanker(ComposerBlock, ABC):
    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from incontext.blocks.chunk_sorting import ChunkSorter
        return ChunkRanker, ChunkSorter


class NegativePathDistanceRanker(ChunkRanker):
    @staticmethod
    def _path_distance(path_from: str, path_to: str) -> int:
        path_from = os.path.normpath(path_from)
        path_to = os.path.normpath(path_to)

        if path_from == path_to:
            warnings.warn(f'Data leakage: the {path_from} completion file is contained in the repo snapshot.')

        divided_path_from = path_from.split(os.path.sep)
        divided_path_to = path_to.split(os.path.sep)

        common_len = 0
        for segment_from, segment_to in zip(divided_path_from, divided_path_to):
            if segment_from == segment_to:
                common_len += 1
            else:
                break

        num_residuals_from = len(divided_path_from) - common_len - 1
        num_residuals_to = len(divided_path_to) - common_len - 1

        return num_residuals_from + num_residuals_to

    def __call__(self, chunks: Sequence[Chunk], datapoint: Datapoint) -> Sequence[Chunk]:
        path_to = datapoint.completion_file['filename']
        for chunk in chunks:
            dist = self._path_distance(chunk.file_ref.metadata['filename'], path_to)
            chunk.rank.append(-dist)
        return chunks


class FileExtensionRanker(ChunkRanker):
    def __init__(self, ordered_groups: list[list[str]]) -> None:
        self.group_weights = {
            extension: weight
            for weight, group in enumerate(ordered_groups)
            for extension in group
        }

    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        for chunk in chunks:
            extension = '.' + chunk.file_ref.metadata['filename'].split('.')[-1]
            chunk.rank.append(self.group_weights.get(extension, -1))
        return chunks


class FunctionCallRanker(ChunkRanker):
    ENCODING = 'utf8'

    def __init__(self, is_relative: bool) -> None:
        self.is_relative = is_relative

        py_language = tree_sitter.Language(tree_sitter_python.language())
        self.parser = tree_sitter.Parser(py_language)

    def dfs_count(self, node: tree_sitter.Node) -> int:
        return (node.type == 'call') + sum(self.dfs_count(child) for child in node.children)

    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        for chunk in chunks:
            if chunk.file_ref.metadata['filename'].endswith('.py'):
                bytecode = bytes(chunk.content, self.ENCODING)
                tree = self.parser.parse(bytecode)
                num_calls = self.dfs_count(tree.root_node)
            else:
                num_calls = 0

            if self.is_relative:
                num_calls /= len(chunk.content)

            chunk.rank.append(num_calls)
        return chunks


class RandomRanker(ChunkRanker):
    def __init__(self, random_seed: int | None) -> None:
        self.generator = random.Random(random_seed)

    def __call__(self, chunks: Sequence[Chunk], _datapoint: Datapoint) -> Sequence[Chunk]:
        ranks = list(range(len(chunks)))
        self.generator.shuffle(ranks)
        for rank, chunk in zip(ranks, chunks):
            chunk.rank.append(rank)
        return chunks


class IoURanker(ChunkRanker):
    def __init__(self, min_len: int) -> None:
        self.min_len = min_len

    def _get_lines(self, content: str) -> set[str]:
        return {line for line in map(str.strip, content.split('\n')) if len(line) >= self.min_len}

    def __call__(self, chunks: Sequence[Chunk], datapoint: Datapoint) -> Sequence[Chunk]:
        target_lines = self._get_lines(datapoint.completion_file['content'])
        for chunk in chunks:
            chunk_lines = self._get_lines(chunk.content)
            iou_score = len(target_lines & chunk_lines) / len(target_lines | chunk_lines)
            chunk.rank.append(iou_score)
        return chunks
