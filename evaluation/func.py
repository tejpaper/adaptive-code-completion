from evaluation.data_structures import LCADatapoint
from incontext import ChainedComposer

import math
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize(text: str,
             tokenizer: PreTrainedTokenizerBase,
             max_seq_len: int,
             num_chars_per_token: int = 6,
             ) -> list[int]:
    num_chars = max_seq_len * num_chars_per_token
    trunc_text = text[-num_chars:]

    tokenized_text = tokenizer(
        text=trunc_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    if len(text) > len(trunc_text) and len(tokenized_text.input_ids) < max_seq_len:
        return tokenize(text, tokenizer, max_seq_len, math.ceil(1.5 * num_chars_per_token))
    else:
        return tokenized_text.input_ids[-max_seq_len:]


@torch.inference_mode()
def evaluate_datapoint(datapoint: LCADatapoint,
                       line_type: Literal['infile', 'inproject'],
                       context_size: int,
                       composer: ChainedComposer,
                       tokenizer: PreTrainedTokenizerBase,
                       model: PreTrainedModel,
                       ) -> tuple[int, int]:
    max_new_tokens = 128
    context_size -= max_new_tokens

    num_matches = 0
    num_lines = 0

    for line_idx in datapoint.completion_lines[line_type]:
        completion_lines = datapoint.completion_file['content'].split('\n')

        composed_datapoint = composer.compose(dict(
            repo=datapoint.repo,
            completion_file=dict(
                filename=datapoint.completion_file['filename'],
                content='\n'.join(completion_lines[:line_idx])
            ),
            repo_snapshot=datapoint.repo_snapshot))
        ground_truth = completion_lines[line_idx]

        tokenized_completion = tokenize(
            text=composed_datapoint['composed_completion'],
            tokenizer=tokenizer,
            max_seq_len=context_size)
        tokenized_context = tokenize(
            text=composed_datapoint['composed_context'],
            tokenizer=tokenizer,
            max_seq_len=context_size - len(tokenized_completion),
        )

        tokenized_input = torch.tensor([tokenized_context + tokenized_completion], device=model.device)

        tokenized_generation = model.generate(
            input_ids=tokenized_input,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            stop_strings='\n',
        )

        full_generation = tokenizer.decode(tokenized_generation[0], skip_special_tokens=True)
        generation = full_generation.split('\n')[-2]

        num_matches += (generation.strip() == ground_truth.strip())
        num_lines += 1

    return num_matches, num_lines
