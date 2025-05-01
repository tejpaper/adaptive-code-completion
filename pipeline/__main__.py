from incontext.init_from_config import find_class
from pipeline.data.dataset import load_dataset
from pipeline.model import init_tokenizer, init_model
from pipeline.outputs.metrics import init_metrics

import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')


@hydra.main(config_path=CONFIGS_DIR, config_name='pipeline', version_base=None)
def main(config: DictConfig) -> None:
    argv_sh = ' \\\n'.join(['python3 -m pipeline'] + sys.argv[1:])

    run_dir = os.path.join(PROJECT_DIR, 'runs', config.run_name)
    argv_sh_file = os.path.join(run_dir, config.argv_sh_file)
    checkpoints_dir = os.path.join(run_dir, config.checkpoints_dir)
    logs_dir = os.path.join(run_dir, config.logs_dir)

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

    adapter_class = find_class(name=adapter_name, module_name='pipeline.model.adapters')
    checkpointer_class = find_class(
        name=checkpointer_name,
        module_name='pipeline.outputs.checkpointers',
        normalization_func={
            'top_k_checkpointer': 'TopKCheckpointManager',
            'checkpointer': 'CheckpointManager',
        }.__getitem__,
    )
    logger_class = find_class(name=logger_name, module_name='pipeline.outputs.loggers')
    preprocessor_class = find_class(name=preprocessor_name, module_name='pipeline.data.preprocessors')
    trainer_class = find_class(name=trainer_name, module_name='pipeline.trainers')

    checkpointer = checkpointer_class(**config.checkpointer, directory=checkpoints_dir)

    tokenizer = init_tokenizer(**config.model)
    model = init_model(**config.model)

    adapter = adapter_class(**config.adapter, model_name=config.model.model_name)

    model = adapter.adapt(model)

    preprocessor = preprocessor_class(**config.preprocessor, tokenizer=tokenizer)

    logger = logger_class(
        **config.logger,
        directory=logs_dir,
        name=config.run_name,
        config=dict(config) | {'config_choices': config_choices},
    )

    train_ds, valid_ds, add_valid_ds = load_dataset(**config.dataset, **config.split)

    train_ds.set_transform(preprocessor)
    valid_ds.set_transform(preprocessor)

    if add_valid_ds is not None:
        add_preprocessor_class = find_class(
            name=os.path.dirname(config.additional_preprocessor),
            module_name='pipeline.data.preprocessors',
        )
        add_preprocessor_config = OmegaConf.load(
            os.path.join(CONFIGS_DIR, f'preprocessor/{config.additional_preprocessor}.yaml'),
        )
        add_preprocessor = add_preprocessor_class(**add_preprocessor_config, tokenizer=tokenizer)
        add_valid_ds.set_transform(add_preprocessor)

    train_metrics = init_metrics(
        loaded_config=config.metrics.train_metrics,
        configs_dir=CONFIGS_DIR,
        tokenizer=tokenizer)
    valid_metrics = init_metrics(
        loaded_config=config.metrics.valid_metrics,
        configs_dir=CONFIGS_DIR,
        tokenizer=tokenizer,
    )

    trainer = trainer_class(
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
