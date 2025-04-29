from incontext.composer.chained_composer import ChainedComposer

import os

import yaml
from omegaconf import OmegaConf


def find_class(name: str, module_name: str) -> type:
    normalized_name = name.replace('_', '').lower()
    module = __import__(module_name)

    for attr_name in dir(module):
        if attr_name.lower() == normalized_name:
            return getattr(module, attr_name)

    raise ValueError('Could not find class matching.')


def init_from_config(path_to_config: str) -> ChainedComposer:
    config_dir = os.path.dirname(path_to_config)
    blocks_dir = os.path.join(config_dir, 'blocks')

    config = OmegaConf.load(path_to_config)
    formatted_config = dict(
        pre_context_prompt=config.pre_context_prompt,
        post_context_prompt=config.post_context_prompt,
        path_comment_template=config.path_comment_template,
        blocks=list(),
    )

    for path in config.block_configs:
        full_path = os.path.join(blocks_dir, path)
        block_name = os.path.basename(os.path.dirname(path))

        with open(full_path) as stream:
            block_config = yaml.safe_load(stream) or dict()

        block = find_class(block_name, 'incontext')(**block_config)
        formatted_config['blocks'].append(block)

    composer = ChainedComposer(**formatted_config)
    return composer
