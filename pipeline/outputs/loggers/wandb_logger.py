from pipeline.outputs.loggers.local_logger import LocalLogger
from pipeline.outputs.loggers.logger_base import Log

import wandb


class WandbLogger(LocalLogger):
    def __init__(self,
                 train_csv: str,
                 valid_csv: str,
                 stdout_file: str,
                 stderr_file: str,
                 directory: str,
                 **wandb_init_kwargs,
                 ) -> None:
        super().__init__(train_csv, valid_csv, stdout_file, stderr_file, directory)
        wandb_init_kwargs['id'] = wandb_init_kwargs.get('id', wandb_init_kwargs['name'])
        wandb.init(**wandb_init_kwargs)

    def log(self, metrics: Log) -> Log:
        wandb_log = dict()
        if 'train_metrics' in metrics:
            wandb_log['train'] = metrics['train_metrics']
        if 'valid_metrics' in metrics:
            wandb_log['validation'] = metrics['valid_metrics']

        wandb.log(wandb_log, step=metrics['iteration_number'])
        return super().log(metrics)
