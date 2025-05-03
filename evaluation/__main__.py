from evaluation.dataset import LongCodeArenaDataset, DataCollator
from evaluation.data_structures import ExactMatchCounter
from incontext import init_from_config as init_composer
from pipeline.model import init_tokenizer, init_model

import json
import os
import sys

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')


@torch.inference_mode()
@hydra.main(config_path=CONFIGS_DIR, config_name='evaluation', version_base=None)
def main(config: DictConfig) -> None:
    output_file = os.path.join(PROJECT_DIR, f'evaluation/outputs/individual/{config.eval_name}.json')
    if os.path.exists(output_file):
        print(f'{config.eval_name} has already been processed. Skipping this evaluation...')
        return

    argv_sh = ' \\\n'.join(['python3 -m evaluation'] + sys.argv[1:])

    if config.path_to_checkpoint is not None:
        config.model.model_name = config.path_to_checkpoint

    composer = init_composer(os.path.join(CONFIGS_DIR, config.path_to_composer_config))

    tokenizer = init_tokenizer(**config.model)
    tokenizer.truncation_side = 'left'

    model = init_model(**config.model)
    model = model.eval().requires_grad_(False)

    crumpled_dataset = load_dataset(
        path='JetBrains-Research/lca-project-level-code-completion',
        name=f'{config.dataset_type}_context',
        split='test',
    )
    dataset = LongCodeArenaDataset(
        crumpled_dataset=crumpled_dataset,
        context_size=config.context_size,
        composer=composer,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(model.device.type == 'cuda'),
        drop_last=False,
        prefetch_factor=config.prefetch_factor,
        pin_memory_device=str(model.device),
        collate_fn=DataCollator(
            tokenizer=tokenizer,
            context_size=config.context_size,
            max_new_tokens=config.max_new_tokens,
        ),
    )

    inproject_em = ExactMatchCounter()
    infile_em = ExactMatchCounter()
    ref_dict = {'inproject': inproject_em, 'infile': infile_em}
    pbar_iter = tqdm(dataloader, desc='Evaluation steps')

    for line_types, input_ids, attn_mask, ground_truths in pbar_iter:
        tokenized_completions = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attn_mask.to(model.device),
            max_new_tokens=config.max_new_tokens,
            tokenizer=tokenizer,
            stop_strings='\n',
        )

        completions = tokenizer.batch_decode([
                tokenized_completion[context_length:]
                for context_length, tokenized_completion in zip(
                    attn_mask.sum(-1), tokenized_completions
                )],
            skip_special_tokens=True,
        )

        for line_type, completion, ground_truth in zip(line_types, completions, ground_truths):
            ref_dict[line_type].num_matches += (completion.strip() == ground_truth.strip())
            ref_dict[line_type].num_lines += 1

        pbar_iter.set_description(
            f'Evaluation steps; inproject={inproject_em.value * 100:.1f}, infile={infile_em.value * 100:.1f}'
        )

    result = OmegaConf.to_container(config)
    result.pop('eval_name')
    result['composer'] = repr(composer)
    result['sh'] = argv_sh
    result['exact_match'] = {
        'inproject': {
            'num_matches': inproject_em.num_matches,
            'num_lines': inproject_em.num_lines,
            'value': inproject_em.value,
        },
        'infile': {
            'num_matches': infile_em.num_matches,
            'num_lines': infile_em.num_lines,
            'value': infile_em.value,
        },
    }

    with open(output_file, 'w') as stream:
        json.dump(result, stream, indent=4)


if __name__ == '__main__':
    main()
