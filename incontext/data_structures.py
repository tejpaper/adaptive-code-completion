from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class File:
    content: str
    metadata: dict[str, Any]


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any]
    file_ref: File
    rank: list = field(default_factory=list)  # of comparable elements


class CompletionFile(TypedDict):
    filename: str
    content: str


class RepoSnapshot(TypedDict):
    filename: list[str]
    content: list[str]


@dataclass
class Datapoint:
    repo: str
    completion_file: CompletionFile
    repo_snapshot: RepoSnapshot


class BatchDatapoint(TypedDict):
    repo: list[str]
    completion_file: list[CompletionFile]
    repo_snapshot: list[RepoSnapshot]


class ComposedDatapoint(TypedDict):
    pre_context_prompt: str
    composed_context: str
    composed_completion: str


class BatchComposedDatapoint(TypedDict):
    pre_context_prompt: list[str]
    composed_context: list[str]
    composed_completion: list[str]
