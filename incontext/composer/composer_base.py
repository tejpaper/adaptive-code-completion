from incontext.blocks.file_preprocessing import NewlinePreprocessor
from incontext.data_structures import (
    BatchComposedDatapoint,
    BatchDatapoint,
    ComposedDatapoint,
    Datapoint,
)

from abc import ABC, abstractmethod
from typing import Any


class ComposerBase(ABC):
    def __init__(self,
                 pre_context_prompt: str,
                 post_context_prompt: str,
                 path_comment_template: str,
                 ) -> None:
        self.pre_context_prompt = pre_context_prompt
        self.post_context_prompt = post_context_prompt
        self.path_comment_template = path_comment_template

    def get_pre_context_prompt(self, datapoint: Datapoint) -> str:
        return self.pre_context_prompt.format(datapoint.repo)

    def get_post_context_prompt(self, _datapoint: Datapoint) -> str:
        return self.post_context_prompt

    @abstractmethod
    def compose_context(self, datapoint: Datapoint) -> str:
        raise NotImplementedError

    def compose_completion(self, datapoint: Datapoint) -> str:
        return self.path_comment_template.format(
            filename=datapoint.completion_file['filename'],
            content=NewlinePreprocessor.unify_newlines(datapoint.completion_file['content']),
        )

    def compose(self, datapoint: dict[str, Any]) -> ComposedDatapoint:
        datapoint = Datapoint(**datapoint)

        return ComposedDatapoint(
            pre_context_prompt=self.get_pre_context_prompt(datapoint),
            composed_context=self.compose_context(datapoint) + self.get_post_context_prompt(datapoint),
            composed_completion=self.compose_completion(datapoint),
        )

    def compose_batch(self, batch: BatchDatapoint) -> BatchComposedDatapoint:
        batch_keys = batch.keys()
        composed_batch_keys = BatchComposedDatapoint.__required_keys__
        # transpose and compose
        batch = [self.compose(dict(zip(batch_keys, data))) for data in zip(*batch.values())]
        # transpose back
        batch = {key: list(map(lambda x: x.get(key), batch)) for key in composed_batch_keys}
        return batch
