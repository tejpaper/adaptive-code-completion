from incontext.init_from_config import find_class
from pipeline.data.dataset import train_test_split
from pipeline.model.init import init_tokenizer_model
from pipeline.outputs.metrics.init import init_metrics

import os
import sys

import hydra
import pandas as pd
from datasets import Dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

LCA_SOLVERS_DIR = os.path.dirname(os.path.dirname(__file__))

# configs
CONFIGS_DIR = os.path.join(LCA_SOLVERS_DIR, 'configs')
MAIN_CONFIG = 'defaults'

# run directory
RUNS_DIR = os.path.join(LCA_SOLVERS_DIR, 'runs')
ARGV_SH_FILE = 'run.sh'
CHECKPOINTS_DIR = 'checkpoints'
LOGS_DIR = 'logs'

# TODO: unify init functions


def reverse_context(string: str) -> str:
    return '<file_sep>'.join(string.split('<file_sep>')[:-1][::-1] + [''])


@hydra.main(config_path=CONFIGS_DIR, config_name=MAIN_CONFIG, version_base=None)
def main(config: DictConfig) -> None:
    argv_sh = ' \\\n'.join(['python3 -m pipeline'] + sys.argv[1:])

    run_dir = os.path.join(RUNS_DIR, config.run_name)
    argv_sh_file = os.path.join(run_dir, ARGV_SH_FILE)
    checkpoints_dir = os.path.join(run_dir, CHECKPOINTS_DIR)
    logs_dir = os.path.join(run_dir, LOGS_DIR)

    if os.path.exists(run_dir):
        with open(argv_sh_file) as stream:
            old_argv_sh = stream.read()

        if argv_sh != old_argv_sh:
            input(f'Mismatch of script arguments with an older instance of the same run ({argv_sh_file}).\n'
                   'Press ENTER to continue with the new ones.')
    else:
        os.mkdir(run_dir)
        os.mkdir(checkpoints_dir)
        os.mkdir(logs_dir)

        os.mknod(os.path.join(checkpoints_dir, '.gitkeep'))
        os.mknod(os.path.join(logs_dir, '.gitkeep'))

    with open(argv_sh_file, 'w') as stream:
        stream.write(argv_sh)

    config_choices = HydraConfig.get().runtime.choices
    adapter_name, checkpointer_name, logger_name, preprocessor_name, trainer_name = [
        os.path.dirname(config_choices.get(cfg_group))
        for cfg_group in ('adapter', 'checkpointer', 'logger', 'preprocessor', 'trainer')
    ]

    adapter_cls = find_class(name=adapter_name, module_name='pipeline.model.adapters')
    checkpointer_cls = find_class(
        name=checkpointer_name,
        module_name='pipeline.outputs.checkpointers',
        normalization_func={
            'top_k_checkpointer': 'TopKCheckpointManager',
            'checkpointer': 'CheckpointManager',
        }.__getitem__,
    )
    logger_cls = find_class(name=logger_name, module_name='pipeline.outputs.loggers')
    preprocessor_cls = find_class(name=preprocessor_name, module_name='pipeline.data.preprocessors')
    trainer_cls = find_class(name=trainer_name, module_name='pipeline.trainers')

    checkpointer = checkpointer_cls(**config.checkpointer, directory=checkpoints_dir)

    tokenizer, model = init_tokenizer_model(config.model)

    adapter = adapter_cls(**config.adapter, model_name=config.model.model_name)

    model = adapter.adapt(model)

    preprocessor = preprocessor_cls(**config.preprocessor, tokenizer=tokenizer)

    logger = logger_cls(
        **config.logger,
        directory=logs_dir,
        name=config.run_name,
        config=dict(config) | {'config_choices': config_choices},
    )

    dataset = pd.read_parquet(config.dataset.main_dataset_path)
    train_ids, test_ids = train_test_split(dataset, **dict(config.split))

    if config.dataset.reversed_context:
        dataset['composed_context'] = dataset.composed_context.apply(reverse_context)
    if config.dataset.file_level:
        dataset['composed_context'] = ''

    train_ds = Dataset.from_pandas(dataset.iloc[train_ids])
    train_ds.set_transform(preprocessor)
    valid_ds = Dataset.from_pandas(dataset.iloc[test_ids])
    valid_ds.set_transform(preprocessor)

    if config.dataset.add_dataset_path is not None:
        assert 'additional_preprocessor' in config

        add_preprocessor_cls = find_class(
            name=os.path.dirname(config.additional_preprocessor),
            module_name='pipeline.data.preprocessors',
        )
        add_preprocessor_config = OmegaConf.load(
            os.path.join(CONFIGS_DIR, f'preprocessor/{config.additional_preprocessor}.yaml'),
        )
        add_preprocessor = add_preprocessor_cls(**add_preprocessor_config, tokenizer=tokenizer)

        add_dataset = pd.read_parquet(config.dataset.add_dataset_path)
        assert list(dataset.pre_context_prompt) == list(add_dataset.pre_context_prompt)
        add_valid_ds = Dataset.from_pandas(add_dataset.iloc[test_ids])
        add_valid_ds.set_transform(add_preprocessor)
    else:
        add_valid_ds = None

    train_metrics = init_metrics(
        loaded_config=config.metrics.train_metrics,
        configs_dir=CONFIGS_DIR,
        tokenizer=tokenizer)
    valid_metrics = init_metrics(
        loaded_config=config.metrics.valid_metrics,
        configs_dir=CONFIGS_DIR,
        tokenizer=tokenizer,
    )

    trainer = trainer_cls(
        **config.trainer,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        add_valid_ds=add_valid_ds,
        adapter=adapter,
        checkpointer=checkpointer,
        logger=logger,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics)
    trainer.train(verbose=True)

    logger.message('Run successfully completed.')


if __name__ == '__main__':
    main()
