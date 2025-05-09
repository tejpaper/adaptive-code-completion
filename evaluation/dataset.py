from evaluation.data_structures import LongCodeArenaDatapoint
from incontext import ChainedComposer
from incontext.data_structures import Datapoint

import math
from dataclasses import asdict
from typing import Literal

import torch
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


LineType = Literal['inproject', 'infile']
EvalSample = tuple[LineType, str, str, str]
EvalBatch = tuple[list[LineType], torch.Tensor, torch.Tensor, list[str]]


class LongCodeArenaDataset(Dataset):
    def __init__(self,
                 crumpled_dataset: HuggingFaceDataset,
                 context_size: int,
                 composer: ChainedComposer,
                 allow_leak: bool,
                 ) -> None:
        self.crumpled_dataset = crumpled_dataset
        self.context_size = context_size
        self.composer = composer
        self.allow_leak = allow_leak

        self.indices = list()
        for datapoint_idx, datapoint in enumerate(crumpled_dataset):
            for line_type in ('inproject', 'infile'):
                for line_idx in datapoint['completion_lines'][line_type]:
                    self.indices.append((datapoint_idx, line_type, line_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> EvalSample:
        datapoint_idx, line_type, line_idx = self.indices[idx]
        datapoint = LongCodeArenaDatapoint(**self.crumpled_dataset[datapoint_idx])

        completion_lines = datapoint.completion_file['content'].split('\n')
        incontext_datapoint = Datapoint(
            repo=datapoint.repo,
            completion_file=datapoint.completion_file,
            repo_snapshot=datapoint.repo_snapshot,
        )

        if self.allow_leak:
            composed_datapoint = self.composer.compose(asdict(incontext_datapoint))
            incontext_datapoint.completion_file['content'] = '\n'.join(completion_lines[:line_idx])
        else:
            incontext_datapoint.completion_file['content'] = '\n'.join(completion_lines[:line_idx])
            composed_datapoint = self.composer.compose(asdict(incontext_datapoint))
        
        composed_datapoint['composed_completion'] = self.composer.compose_completion(incontext_datapoint)
        ground_truth = completion_lines[line_idx]

        return (
            line_type,
            composed_datapoint['composed_context'] + ('\n', '')[composed_datapoint['composed_context'].endswith('\n')],
            composed_datapoint['composed_completion'] + ('\n', '')[composed_datapoint['composed_completion'].endswith('\n')],
            ground_truth + ('\n', '')[ground_truth.endswith('\n')],
        )


class DataCollator:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 context_size: int,
                 ) -> None:
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.no_bos = (tokenizer.bos_token_id is None)

    def _tokenize(self, 
                  text: str,
                  max_seq_len: int,
                  num_chars_per_token: int = 6,
                  ) -> list[int]:
        if max_seq_len == 0:
            return []

        num_chars = max_seq_len * num_chars_per_token
        trunc_text = text[-num_chars:]

        tokenized_text = self.tokenizer(
            text=trunc_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        if len(text) > len(trunc_text) and len(tokenized_text.input_ids) < max_seq_len:
            return self._tokenize(text, max_seq_len, math.ceil(1.5 * num_chars_per_token))
        else:
            return tokenized_text.input_ids[-max_seq_len:]

    def __call__(self, batch: list[EvalSample]) -> EvalBatch:
        line_types = list()
        input_ids = list()
        ground_truths = list()

        for line_type, repo_context, file_prefix, ground_truth in batch:
            tokenized_completion = self._tokenize(
                text=file_prefix + ground_truth,
                max_seq_len=self.context_size + self.no_bos)
            tokenized_repo_context = self._tokenize(
                text=repo_context,
                max_seq_len=self.context_size - len(tokenized_completion) + self.no_bos,
            )
            
            tokenized_input = tokenized_repo_context + tokenized_completion
            if not self.no_bos:
                tokenized_input = [self.tokenizer.bos_token_id] + tokenized_input
            tokenized_input = tokenized_input[:-1]

            line_types.append(line_type)
            input_ids.append(tokenized_input)
            ground_truths.append(ground_truth)

        padded_batch = self.tokenizer.pad(
            encoded_inputs={'input_ids': input_ids},
            padding='longest',
            return_attention_mask=True,
            return_tensors='pt')
        assert padded_batch.input_ids.shape[-1] <= self.context_size, padded_batch.input_ids.shape

        return line_types, padded_batch.input_ids, padded_batch.attention_mask, ground_truths
