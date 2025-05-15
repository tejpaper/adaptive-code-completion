from incontext.blocks.block import ComposerBlock
from incontext.data_structures import Datapoint

import random
from abc import ABC
from typing import Type

from transformers import AutoTokenizer


class ContextPostprocessor(ComposerBlock, ABC):
    last_block_permit = True

    @property
    def next_blocks(self) -> tuple[Type[ComposerBlock], ...]:
        return ContextPostprocessor,


class PartialMemoryPostprocessor(ContextPostprocessor):
    def __init__(self, dropout: float, random_seed: int | None) -> None:
        if not 0 <= dropout <= 1:
            raise ValueError('dropout must be selected from the interval [0, 1]. '
                             f'Got {dropout} instead.')
        self.dropout = dropout
        self.generator = random.Random(random_seed)

    def __call__(self, context: str, _datapoint: Datapoint) -> str:
        # dropping of path comments can occasionally happen
        return '\n'.join(line for line in context.split('\n') if self.generator.random() >= self.dropout)


class LineLengthPostprocessor(ContextPostprocessor):
    def __init__(self, min_len: int, max_len: int) -> None:
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, context: str, _datapoint: Datapoint) -> str:
        return '\n'.join(line for line in context.split('\n') if self.min_len <= len(line) <= self.max_len)


class LineStripPostprocessor(ContextPostprocessor):
    def __call__(self, context: str, _datapoint: Datapoint) -> str:
        return '\n'.join(line.strip() for line in context.split('\n'))


class CompletionLeakPostprocessor(ContextPostprocessor):
    def __init__(self,
                 chars_lower_bound: int,
                 context_size: int,
                 num_segments: int,
                 tokenizer_name: str,
                 trust_remote_code: bool,
                 random_seed: int | None,
                 ) -> None:
        self.chars_lower_bound = chars_lower_bound
        self.context_size = context_size
        self.num_segments = num_segments

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        self.generator = random.Random(random_seed)

    def _select_random_subsequences(self, segment_lens: list[int], num_context_lines: int) -> list[tuple[int, int]]:
        available = [True] * num_context_lines
        subsequences = list()

        for length in segment_lens:
            valid_starts = [
                i for i in range(num_context_lines - length + 1)
                if all(available[i:i + length])]

            if not valid_starts:
                subsequences.append((0, 0))
                continue

            start = self.generator.choice(valid_starts)
            end = start + length

            for i in range(start, end):
                available[i] = False

            subsequences.append((start, end))

        return subsequences

    def _leak_completion(self, context: str, completion: str) -> str:
        completion_lines = [line + '\n' for line in completion.rstrip().split('\n')]

        num_segments = min(len(completion_lines), self.num_segments)
        breaks = sorted(self.generator.sample(range(len(completion_lines)), k=(num_segments - 1)))

        segments = list()
        for start_idx, end_idx in zip([0] + breaks, breaks + [len(completion_lines)]):
            segments.append((start_idx, end_idx))
        self.generator.shuffle(segments)

        segment_lens = [end_idx - start_idx for start_idx, end_idx in segments]

        context_lines = [line + '\n' for line in context.rstrip().split('\n')]

        while len(context_lines) < sum(segment_lens):
            segments.pop()
            segment_lens.pop()
        assert len(context_lines) >= sum(segment_lens)

        replace_positions = self._select_random_subsequences(segment_lens, len(context_lines))

        for (cmp_start, cmp_end), (ctx_start, ctx_end) in zip(segments, replace_positions):
            if ctx_start >= ctx_end:
                continue

            for offset in range(cmp_end - cmp_start):
                context_lines[ctx_start + offset] = completion_lines[cmp_start + offset]

        return ''.join(context_lines)

    def __call__(self, context: str, datapoint: Datapoint) -> str:
        tokenized_context = self.tokenizer(
            context[-self.chars_lower_bound:],
            return_attention_mask=False,
        ).input_ids[-self.context_size:]
        num_required_chars = len(self.tokenizer.decode(tokenized_context))

        visible_context = context[-num_required_chars:]
        completion = datapoint.completion_file['content']
        contaminated_context = self._leak_completion(visible_context, completion)

        return contaminated_context


# hardcode to make init_from_config clearer
class DseekCompletionLeakPostprocessor(CompletionLeakPostprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs,
                         tokenizer_name='deepseek-ai/deepseek-coder-1.3b-base',
                         trust_remote_code=True)


# hardcode to make init_from_config clearer
class OCoderCompletionLeakPostprocessor(CompletionLeakPostprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs,
                         tokenizer_name='infly/OpenCoder-1.5B-Base',
                         trust_remote_code=True)


class ReversedContextPostprocessor(ContextPostprocessor):
    def __init__(self,
                 chars_lower_bound: int,
                 context_size: int,
                 chunks_sep: str,
                 tokenizer_name: str,
                 trust_remote_code: bool,
                 ) -> None:
        self.chars_lower_bound = chars_lower_bound
        self.context_size = context_size
        self.chunks_sep = chunks_sep

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name,
            trust_remote_code=trust_remote_code,
        )

    def __call__(self, context: str, datapoint: Datapoint) -> str:
        completion = datapoint.completion_file['content']

        tokenized_context = self.tokenizer(
            (context + completion)[-self.chars_lower_bound:],
            return_attention_mask=False,
        ).input_ids[-self.context_size:]
        num_required_chars = len(self.tokenizer.decode(tokenized_context)) - len(completion)

        visible_context = context[-num_required_chars:]
        reversed_context = self.chunks_sep.join(visible_context.split(self.chunks_sep)[::-1])

        return reversed_context


# hardcode to make init_from_config clearer
class OCoderReversedContextPostprocessor(ReversedContextPostprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs,
                         tokenizer_name='infly/OpenCoder-1.5B-Base',
                         trust_remote_code=True)


class RandomTokensPostprocessor(ContextPostprocessor):
    def __init__(self,
                 context_size: int,
                 tokenizer_name: str,
                 trust_remote_code: bool,
                 random_seed: int | None,
                 ) -> None:
        self.context_size = context_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        self.generator = random.Random(random_seed)

        special_token_ids = set(list(self.tokenizer.get_added_vocab().values()) + self.tokenizer.all_special_ids)
        self.allowed_token_ids = list(set(range(len(self.tokenizer))).difference(special_token_ids))

    def __call__(self, _context: str, _datapoint: Datapoint) -> str:
        return self.tokenizer.decode(self.generator.choices(self.allowed_token_ids, k=self.context_size))


# hardcode to make init_from_config clearer
class DseekRandomTokensPostprocessor(RandomTokensPostprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs,
                         tokenizer_name='deepseek-ai/deepseek-coder-1.3b-base',
                         trust_remote_code=True)


# hardcode to make init_from_config clearer
class OCoderRandomTokensPostprocessor(RandomTokensPostprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs,
                         tokenizer_name='infly/OpenCoder-1.5B-Base',
                         trust_remote_code=True)
