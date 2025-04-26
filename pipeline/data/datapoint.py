from dataclasses import dataclass
from typing import TypedDict


class CompletionFile(TypedDict):
    filename: str
    content: str


class RepoSnapshot(TypedDict):
    filename: list[str]
    content: list[str]


@dataclass
class Datapoint:
    repo: str
    commit_hash: str
    completion_file: CompletionFile
    repo_snapshot: RepoSnapshot


class BatchDatapoint(TypedDict):
    repo: list[str]
    commit_hash: list[str]
    completion_file: list[CompletionFile]
    repo_snapshot: list[RepoSnapshot]
