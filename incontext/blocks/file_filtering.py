from incontext.blocks.block import ComposerBlock
from incontext.data_structures import Datapoint, File

from abc import ABC
from typing import Sequence, Type


class FileFilter(ComposerBlock, ABC):
    first_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from incontext.blocks.file_chunking import FileChunker
        from incontext.blocks.file_preprocessing import FilePreprocessor
        return FileFilter, FilePreprocessor, FileChunker


class NullFileFilter(FileFilter):
    def __call__(self, *_args, **_kwargs) -> Sequence[File]:
        return []


class InclusiveFileExtensionFilter(FileFilter):
    def __init__(self, whitelist: list[str]) -> None:
        self.whitelist = tuple(whitelist)

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        return [file for file in files if file.metadata['filename'].endswith(self.whitelist)]


class ExclusiveFileExtensionFilter(FileFilter):
    def __init__(self, blacklist: list[str]) -> None:
        self.blacklist = tuple(blacklist)

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        return [file for file in files if not file.metadata['filename'].endswith(self.blacklist)]


class EmptyFileFilter(FileFilter):
    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        return [file for file in files if file.content.strip()]


class FileLengthFilter(FileFilter):
    def __init__(self, min_len: int, max_len: int) -> None:
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, files: Sequence[File], _datapoint: Datapoint) -> Sequence[File]:
        return [file for file in files if self.min_len <= len(file.content) <= self.max_len]
