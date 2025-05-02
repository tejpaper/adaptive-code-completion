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
class LongCodeArenaDatapoint:
    repo: str
    commit_hash: str
    completion_file: CompletionFile
    completion_lines: CompletionLines
    repo_snapshot: RepoSnapshot
    completion_lines_raw: CompletionLines | None = None


@dataclass
class ExactMatchCounter:
    num_matches: int = 0
    num_lines: int = 0

    @property
    def value(self) -> float:
        if self.num_lines > 0:
            return self.num_matches / self.num_lines
        else:
            return 0
