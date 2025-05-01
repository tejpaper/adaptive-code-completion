from incontext.data_structures import CompletionFile, RepoSnapshot

from dataclasses import dataclass
from typing import TypedDict


class CompletionLines(TypedDict, total=False):
    commited: list[int]
    common: list[int]
    infile: list[int]
    inproject: list[int]
    non_informative: list[int]
    random: list[int]
    other: list[int]


@dataclass
class LCADatapoint:
    repo: str
    commit_hash: str
    completion_file: CompletionFile
    completion_lines: CompletionLines
    repo_snapshot: RepoSnapshot
    completion_lines_raw: CompletionLines | None = None
