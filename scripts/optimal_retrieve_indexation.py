from pipeline.data.datapoint import CompletionLines, RepoSnapshot, Datapoint
from pipeline.data.preprocessors.completion_loss_preprocessor import CompletionLossPreprocessor
from pipeline.environment.hardware import get_free_device
from pipeline.outputs.metrics.cross_entropy import CrossEntropy
from pipeline.outputs.metrics.exact_match import ExactMatch
from pipeline.outputs.metrics.metric_base import MaskType

import json
import os
import sys
from collections import defaultdict
from typing import TypedDict

import jsonlines
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datasets import Dataset as HuggingFaceDataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithPast

CONFIGS_DIR = 'scripts/configs/optimal_retrieve_indexation'
OUTPUTS_DIR = 'scripts/outputs/optimal_retrieve_indexation'
RAW_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, 'raw')
PREFETCH_FACTOR = 8
INPUTS_TRACEBACK = [None for _ in range(PREFETCH_FACTOR)]


class RawBatch(TypedDict):
    tokenized_sample: list[list[int]]
    completion_lines: list[CompletionLines]
    completion_length: list[int]
    offset_mapping: list[list[tuple[int, int]]]
    newline_positions: list[list[int]]

    anchor_seq_len: int

    repo: list[str]
    commit_hash: list[str]
    completion_filename: list[str]
    context_filename: list[str]


class Batch(TypedDict):
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    completion_mask: torch.Tensor
    category_ids: torch.Tensor
    attention_mask: torch.Tensor

    repo: list[str]
    commit_hash: list[str]
    completion_filename: list[str]
    context_filename: list[str]


class Buffer:
    OPTIMAL_BATCH_SIZE = {
        1024: 40, 1536: 36, 2048: 32, 2560: 34, 3072: 28, 3584: 38,
        4096: 16, 4608: 18, 5120: 30, 5632: 38, 6144: 36, 6656: 34,
        7168: 32, 7680: 28, 8192: 24, 9216: 18, 10240: 22, 11264: 20,
        12288: 18, 13312: 14, 14336: 14, 15360: 12, 16384: 14,
    }

    def __init__(self) -> None:
        self.storage = defaultdict(lambda: defaultdict(list))
        self.last_updated_queue = None

    @property
    def empty(self) -> bool:
        return not self.storage

    def push(self, tokenized_sample: list[int], **kwargs) -> None:
        seq_len = len(tokenized_sample) - 1
        anchor_seq_len = min(self.OPTIMAL_BATCH_SIZE, key=lambda x: (x < seq_len, x))

        kwargs['tokenized_sample'] = tokenized_sample
        for k, v in kwargs.items():
            self.storage[anchor_seq_len][k].append(v)

        self.last_updated_queue = anchor_seq_len

    def pop(self) -> RawBatch | None:
        if self.last_updated_queue is None:
            return None

        batch_size = len(self.storage[self.last_updated_queue]['tokenized_sample'])
        optimal_batch_size = self.OPTIMAL_BATCH_SIZE[self.last_updated_queue]
        assert batch_size <= optimal_batch_size

        if batch_size < optimal_batch_size:
            return None

        raw_batch = self.storage.pop(self.last_updated_queue)
        raw_batch['anchor_seq_len'] = self.last_updated_queue
        self.last_updated_queue = None

        return raw_batch

    def force_pop(self) -> RawBatch:
        if self.empty:
            raise RuntimeError('Buffer is empty.')

        anchor_seq_len = min(self.storage)
        raw_batch = self.storage.pop(anchor_seq_len)
        raw_batch['anchor_seq_len'] = anchor_seq_len

        return raw_batch


class Dataset(IterableDataset, CompletionLossPreprocessor):
    def __init__(self, hf_dataset: HuggingFaceDataset, *args, **kwargs) -> None:
        CompletionLossPreprocessor.__init__(self, *args, **kwargs)
        self.hf_dataset = hf_dataset
        self.buffer = Buffer()

    @staticmethod
    def filter_snapshot(snapshot: RepoSnapshot, completion_filename: str) -> RepoSnapshot:
        path_comment = f'# {completion_filename}\n'
        filtered_snapshot = {'filename': [''], 'content': [path_comment]}

        for context_filename, context_content in zip(*snapshot.values()):
            if (not context_filename.endswith('.py') or
                    not context_content.strip() or
                    context_filename == completion_filename):
                continue

            context_content = context_content.rstrip() + '\n\n' + path_comment
            filtered_snapshot['filename'].append(context_filename)
            filtered_snapshot['content'].append(context_content)

        return filtered_snapshot

    def push_sample(self,
                    sample: Datapoint,
                    context_filename: str,
                    context_content: str,
                    ) -> None:
        tokenized_completion = self.tokenize_composed_completion([sample.completion_file['content']])
        tokenized_contexts = self.tokenize_composed_context([context_content])

        completion = tokenized_completion.input_ids[0]
        context = tokenized_contexts.input_ids[0]

        _, context_len, completion_len = self.calc_lens([], context, completion)
        context = context[-context_len:]
        completion = completion[:completion_len]

        self.buffer.push(
            tokenized_sample=[self.tokenizer.bos_token_id] + context + completion,
            completion_lines=sample.completion_lines,
            completion_length=len(completion),
            offset_mapping=tokenized_completion.offset_mapping[0][:completion_len],
            newline_positions=tokenized_completion.newline_positions[0],
            repo=sample.repo,
            commit_hash=sample.commit_hash,
            completion_filename=sample.completion_file['filename'],
            context_filename=context_filename,
        )

    def collate_batch(self,
                      tokenized_sample: list[list[int]],
                      completion_lines: list[CompletionLines],
                      completion_length: list[int],
                      offset_mapping: list[list[tuple[int, int]]],
                      newline_positions: list[list[int]],
                      anchor_seq_len: int,
                      **kwargs,
                      ) -> Batch:
        padded_batch = self.tokenizer.pad(
            encoded_inputs={'input_ids': tokenized_sample},
            padding='max_length',
            max_length=anchor_seq_len + 1,
            return_attention_mask=True,
            return_tensors='pt')
        input_attn_mask = padded_batch.attention_mask[:, :-1]
        target_attn_mask = padded_batch.attention_mask[:, 1:]

        tokenized_completions_filler = BatchEncoding({
            'length': torch.tensor(completion_length),
            'offset_mapping': offset_mapping,
            'newline_positions': newline_positions,
        })

        return Batch(
            input_ids=padded_batch.input_ids[:, :-1],
            target_ids=padded_batch.input_ids[:, 1:],
            completion_mask=self.get_completion_mask(tokenized_completions_filler, target_attn_mask),
            category_ids=self.get_category_ids(tokenized_completions_filler, completion_lines, target_attn_mask),
            attention_mask=input_attn_mask,
            **kwargs,
        )

    def __iter__(self) -> Batch:
        for sample in tqdm(self.hf_dataset):
            sample = Datapoint(**sample)
            snapshot = self.filter_snapshot(sample.repo_snapshot, sample.completion_file['filename'])

            for context_filename, context_content in zip(*snapshot.values()):
                self.push_sample(sample, context_filename, context_content)

                raw_batch = self.buffer.pop()
                if raw_batch is not None:
                    yield self.collate_batch(**raw_batch)

        while not self.buffer.empty:
            yield self.collate_batch(**self.buffer.force_pop())


@torch.inference_mode
def main() -> None:
    config_filename = sys.argv[1]
    config = OmegaConf.load(os.path.join(CONFIGS_DIR, config_filename))
    session_name = config_filename.split('.')[0]
    output_file = os.path.join(RAW_OUTPUTS_DIR, f'{session_name}.jsonl')

    device = get_free_device()
    dtype = getattr(torch, config.model.dtype)
    torch.set_float32_matmul_precision(config.model.fp32_matmul_precision)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model.model_name,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=config.model.attention,
        use_cache=False)
    model = model.eval().requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model.model_name,
        trust_remote_code=True)
    tokenizer.padding_side = 'right'

    hf_dataset = load_dataset(**config.dataset)
    hf_dataset = hf_dataset.select(range(config.chunk.start, config.chunk.stop))

    dataloader = DataLoader(
        dataset=Dataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            num_chars_per_token=6,
            use_sep_token=False,
            padding=True,
            verbose=False,
            **config.preprocessor,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: x[0],
        pin_memory=True,
        drop_last=False,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory_device=str(device),
    )

    metrics = {
        'completion_exact_match': ExactMatch(tokenizer, 3, MaskType.COMPLETION),
        'infile_exact_match': ExactMatch(tokenizer, 3, MaskType.INFILE),
        'inproject_exact_match': ExactMatch(tokenizer, 3, MaskType.INPROJECT),

        'completion_cross_entropy': CrossEntropy(MaskType.COMPLETION),
        'infile_cross_entropy': CrossEntropy(MaskType.INFILE),
        'inproject_cross_entropy': CrossEntropy(MaskType.INPROJECT),
    }

    with jsonlines.open(output_file, mode='a') as writer:
        for batch in dataloader:
            with device:
                torch.cuda.empty_cache()

            (input_ids, target_ids,
             completion_mask, category_ids, attention_mask,
             repo, commit_hash, completion_filename, context_filename,
             ) = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch.values())

            INPUTS_TRACEBACK.pop(0)
            INPUTS_TRACEBACK.append(tuple(input_ids.shape))

            model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_per_token = F.cross_entropy(
                input=model_output.logits.flatten(0, 1),
                target=target_ids.flatten(0, 1),
                reduction='none',
            ).view_as(target_ids)

            for sample_idx in range(input_ids.shape[0]):
                results = dict()

                for metric_name, metric in metrics.items():
                    metric.micro_batch_update(
                        model_output=CausalLMOutputWithPast(logits=model_output.logits[sample_idx]),
                        target_ids=target_ids[sample_idx],
                        loss_per_token=loss_per_token[sample_idx],
                        completion_mask=completion_mask[sample_idx],
                        category_ids=category_ids[sample_idx],
                    )
                    metric_value = metric.batch_commit()
                    results[metric_name] = metric_value

                writer.write({
                    'repo': repo[sample_idx],
                    'commit_hash': commit_hash[sample_idx],
                    'completion_filename': completion_filename[sample_idx],
                    'results': {
                        context_filename[sample_idx]: results,
                    }
                })


if __name__ == '__main__':
    # Estimated number of samples: 7M
    # Estimated number of tokens: 35B
    # Estimated runtime: 6+ days
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print(f'OOM: {INPUTS_TRACEBACK}')


# appendix
def merge_chunks() -> None:
    indices = dict()

    for chunk_filename in tqdm(os.listdir(RAW_OUTPUTS_DIR)):
        with jsonlines.open(os.path.join(RAW_OUTPUTS_DIR, chunk_filename)) as stream:
            for line in stream:
                repo = line['repo']
                commit_hash = line['commit_hash']
                completion_filename = line['completion_filename']
                results = line['results']

                repo_dict = indices.get(repo, dict())
                commit_dict = repo_dict.get(commit_hash, dict())
                completion_dict = commit_dict.get(completion_filename, dict())

                for content_filename, metrics in results.items():
                    completion_dict[content_filename] = metrics

                commit_dict[completion_filename] = completion_dict
                repo_dict[commit_hash] = commit_dict
                indices[repo] = repo_dict

    with open(os.path.join(OUTPUTS_DIR, 'indices.json'), 'w') as stream:
        json.dump(indices, stream)
