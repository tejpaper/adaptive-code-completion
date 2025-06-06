from incontext.blocks.block import ComposerBlock
from incontext.blocks.file_preprocessing import NewlinePreprocessor
from incontext.data_structures import Chunk, Datapoint, File

import math
from abc import ABC
from enum import Enum
from string import whitespace
from typing import Callable, NamedTuple, Sequence, TypeVar, Type

import tree_sitter
import tree_sitter_python

T = TypeVar('T')


class FileChunker(ComposerBlock, ABC):
    first_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        from incontext.blocks.chunk_assembling import ChunkAssembler
        from incontext.blocks.chunk_ranking import ChunkRanker
        return ChunkRanker, ChunkAssembler


class FileGrainedChunker(FileChunker):  # identity chunker
    def __call__(self, files: Sequence[File], datapoint: Datapoint) -> Sequence[Chunk]:
        chunks = list()
        for file in files:
            assert file.metadata['filename'] != datapoint.completion_file['filename']
            chunks.append(Chunk(
                content=file.content,
                metadata=file.metadata,
                file_ref=file,
            ))

        return chunks


class CodeSegment(str, Enum):
    COMMENT = 'comment_segment'
    DOCSTRING = 'docstring_segment'
    IMPORT = 'import_segment'
    CODE = 'code_segment'
    UNDEFINED = 'undefined_segment'

    @classmethod
    def from_node(cls: Type[T], node: tree_sitter.Node) -> T:
        if 'comment' in node.type.lower():
            return cls.COMMENT
        elif node.type == 'string' and node.text.startswith(CodeSegmentGrainedChunker.DOCSTRING_PREFIX):
            return cls.DOCSTRING
        elif 'import' in node.type.lower():
            return cls.IMPORT
        elif node.child_count == 0:
            return cls.CODE
        else:
            return cls.UNDEFINED


class Segment(NamedTuple):
    start_byte: int
    type: CodeSegment


class CodeSegmentGrainedChunker(FileChunker):
    ENCODING = 'utf8'
    DOCSTRING_PREFIX = (bytes("'''", ENCODING), bytes('"""', ENCODING))

    def __init__(self) -> None:
        py_language = tree_sitter.Language(tree_sitter_python.language())
        self.parser = tree_sitter.Parser(py_language)

    @staticmethod
    def dfs_segmentation(root_node: tree_sitter.Node) -> list[Segment]:
        segments = list()
        queue = [root_node]

        while queue:
            node = queue.pop()
            segment_type = CodeSegment.from_node(node)

            if segment_type != CodeSegment.UNDEFINED:
                if len(segments) == 0 or segment_type != segments[-1].type:
                    segments.append(Segment(node.start_byte, segment_type))
            else:
                queue.extend(reversed(node.children))

        return segments

    @staticmethod
    def strip_lines(string: str, strip_func: Callable[[str], str]) -> str:
        return '\n'.join(map(strip_func, string.split('\n')))

    def __call__(self, files: Sequence[File], datapoint: Datapoint) -> Sequence[Chunk]:
        chunks = list()

        for file in files:
            if not file.metadata['filename'].endswith('.py'):
                chunks.append(Chunk(
                    content=file.content,
                    metadata=file.metadata | {'segment_type': CodeSegment.UNDEFINED},
                    file_ref=file,
                ))
                continue

            assert file.metadata['filename'] != datapoint.completion_file['filename']

            bytecode = bytes(file.content, self.ENCODING)
            tree = self.parser.parse(bytecode)
            segments = self.dfs_segmentation(tree.root_node)

            dummy_segment = Segment(len(bytecode), CodeSegment.UNDEFINED)
            segments.append(dummy_segment)

            comments_chunk = Chunk(
                content='', metadata=file.metadata | {'segment_type': CodeSegment.COMMENT}, file_ref=file)
            docstrings_chunk = Chunk(
                content='', metadata=file.metadata | {'segment_type': CodeSegment.DOCSTRING}, file_ref=file)
            imports_chunk = Chunk(
                content='', metadata=file.metadata | {'segment_type': CodeSegment.IMPORT}, file_ref=file)
            code_chunk = Chunk(
                content='', metadata=file.metadata | {'segment_type': CodeSegment.CODE}, file_ref=file)
            prev_edited_chunk = None

            for i in range(len(segments) - 1):
                start = segments[i].start_byte
                end = segments[i + 1].start_byte
                segment_str = bytecode[start:end].decode(self.ENCODING)
                segment_type = segments[i].type

                match segment_type:
                    case CodeSegment.COMMENT:
                        # inline comment newline fix
                        if prev_edited_chunk is not None and not prev_edited_chunk.content.rstrip(
                                whitespace.replace('\n', '')).endswith('\n'):
                            prev_edited_chunk.content += '\n' + segment_str.split('\n')[-1]

                        if segment_str.count('\n') >= 2:
                            comments_chunk.content += self.strip_lines(segment_str, str.strip)
                            prev_edited_chunk = comments_chunk

                    case CodeSegment.DOCSTRING:
                        docstrings_chunk.content += self.strip_lines(segment_str, str.strip)
                        prev_edited_chunk = docstrings_chunk

                    case CodeSegment.IMPORT:
                        imports_chunk.content += segment_str
                        prev_edited_chunk = imports_chunk

                    case CodeSegment.CODE:
                        code_chunk.content += segment_str
                        prev_edited_chunk = code_chunk

                    case _:
                        raise RuntimeError

            for chunk in (comments_chunk, docstrings_chunk, imports_chunk, code_chunk):
                if chunk.content.strip():
                    chunk.content = self.strip_lines(chunk.content.rstrip(), str.rstrip)
                    chunks.append(chunk)

        return chunks


class DocstringAndCommentOnlyChunker(CodeSegmentGrainedChunker):
    def __call__(self, *args, **kwargs) -> Sequence[Chunk]:
        return [chunk for chunk in super().__call__(*args, **kwargs) if
                chunk.metadata['segment_type'] == CodeSegment.DOCSTRING or
                chunk.metadata['segment_type'] == CodeSegment.COMMENT]


class CodeOnlyChunker(CodeSegmentGrainedChunker):
    def __call__(self, *args, **kwargs) -> Sequence[Chunk]:
        return [chunk for chunk in super().__call__(*args, **kwargs)
                if chunk.metadata['segment_type'] == CodeSegment.CODE]


class FixedLineChunker(FileChunker):
    def __init__(self, chunk_lines_size: int, overlap_lines_size: int) -> None:
        if chunk_lines_size <= overlap_lines_size:
            raise ValueError('chunk_lines_size must be greater than overlap_lines_size.')

        self.chunk_lines_size = chunk_lines_size
        self.overlap_lines_size = overlap_lines_size

    def __call__(self, files: Sequence[File], datapoint: Datapoint) -> Sequence[Chunk]:
        chunks = list()

        for file in files:
            assert file.metadata['filename'] != datapoint.completion_file['filename']

            lines = file.content.split('\n')
            total_lines = len(lines)

            stride = self.chunk_lines_size - self.overlap_lines_size
            for i in range(total_lines, self.chunk_lines_size - 1, -stride):
                start_idx = max(0, i - self.chunk_lines_size)
                chunks.append(Chunk(
                    content='\n'.join(lines[start_idx:i]),
                    metadata=file.metadata,
                    file_ref=file
                ))

            if total_lines % stride != 0:
                chunks.append(Chunk(
                    content='\n'.join(lines[:self.chunk_lines_size]),
                    metadata=file.metadata,
                    file_ref=file
                ))

        return chunks


class CompletionDuplicationChunker(FileChunker):
    def __init__(self, chars_lower_bound: int) -> None:
        self.chars_lower_bound = chars_lower_bound

    def __call__(self, _files: Sequence[File], datapoint: Datapoint) -> Sequence[Chunk]:
        filename = datapoint.completion_file['filename']
        content = NewlinePreprocessor.unify_newlines(datapoint.completion_file['content'])
        num_repeats = math.ceil(self.chars_lower_bound / len(content))

        return [Chunk(
            content=content,
            metadata=dict(filename=filename),
            file_ref=File(
                content=content,
                metadata=dict(filename=filename),
            )) for _ in range(num_repeats)]
