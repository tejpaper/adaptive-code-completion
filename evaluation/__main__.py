from evaluation.data_structures import LCADatapoint
from evaluation.func import evaluate_datapoint
from incontext import init_from_config as init_composer
from pipeline.model import init_tokenizer, init_model

import json
import os
import sys

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')


@hydra.main(config_path=CONFIGS_DIR, config_name='evaluation', version_base=None)
def main(config: DictConfig) -> None:
    argv_sh = ' \\\n'.join(['python3 -m evaluation'] + sys.argv[1:])

    dataset_type = f'{config.dataset_type}_context'
    output_file = os.path.join(PROJECT_DIR, f'evaluation/outputs/individual/{config.eval_name}.json')

    if config.path_to_checkpoint is not None:
        config.model.model_name = config.path_to_checkpoint

    composer = init_composer(os.path.join(CONFIGS_DIR, config.path_to_composer_config))
    
    tokenizer = init_tokenizer(**config.model)
    tokenizer.truncation_side = 'left'
    
    model = init_model(**config.model)
    model = model.eval().requires_grad_(False)

    benchmark_ds = load_dataset(
        path='JetBrains-Research/lca-project-level-code-completion',
        name=dataset_type,
        split='test',
    )

    result = OmegaConf.to_container(config)
    result.pop('eval_name')
    result['composer'] = repr(composer)
    result['sh'] = argv_sh
    result['exact_match'] = {
        'infile': {
            'num_matches': 0,
            'num_lines': 0,
        },
        'inproject': {
            'num_matches': 0,
            'num_lines': 0,
        },
    }

    for datapoint in tqdm(benchmark_ds, desc='Evaluation steps'):
        for line_type in ('infile', 'inproject'):
            num_matches, num_lines = evaluate_datapoint(
                datapoint=LCADatapoint(**datapoint),
                line_type=line_type,
                context_size=config.context_size,
                composer=composer,
                tokenizer=tokenizer,
                model=model,
            )

            result['exact_match'][line_type]['num_matches'] += num_matches
            result['exact_match'][line_type]['num_lines'] += num_lines

    with open(output_file, 'w') as stream:
        json.dump(result, stream, indent=4)


if __name__ == '__main__':
    main()
