from pipeline.outputs.checkpointers.checkpoint import Checkpoint
from pipeline.outputs.loggers.logger_base import Log
from pipeline.outputs.metrics.metric_base import OptimizationMode
from pipeline.outputs.metrics.statistic_base import StatisticName, StatisticValue
from pipeline.outputs.metrics.metrics_registry import METRICS_REGISTRY

import json
import os
import warnings

import torch


class CheckpointManager:  # aka checkpointer
    def __init__(self,
                 main_metric: StatisticName,
                 directory: str,
                 checkpoint_directory_template: str = '{iteration_number:04d}',
                 model_subdirectory: str = 'model',
                 optim_state_filename: str = 'optim.pt',
                 metrics_filename: str = 'metrics.json',  # should be .json
                 ) -> None:
        self.main_metric_name = main_metric
        self.main_metric = METRICS_REGISTRY[main_metric]
        self.directory = directory

        self._checkpoint_directory_template = checkpoint_directory_template
        self._model_subdirectory = model_subdirectory
        self._optim_state_filename = optim_state_filename
        self._metrics_filename = metrics_filename

    def load_metrics(self, checkpoint_dir: str) -> Log:
        metrics_file = os.path.join(checkpoint_dir, self._metrics_filename)
        with open(metrics_file) as stream:
            return Log(**json.load(stream))

    def get_checkpoint_score(self, checkpoint_dir: str) -> StatisticValue:
        checkpoint_dir = os.path.join(self.directory, checkpoint_dir)
        metrics = self.load_metrics(checkpoint_dir)
        metric_value = metrics.get('valid_metrics', metrics['train_metrics']).get(self.main_metric_name)

        if metric_value is None:
            raise RuntimeError(f'The {checkpoint_dir} does not contain information '
                               'about the specified main_metric.')
        elif self.main_metric.mode == OptimizationMode.MIN:
            return metric_value
        else:
            return -metric_value

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        checkpoint_dir = os.path.join(
            self.directory,
            self._checkpoint_directory_template.format(
                iteration_number=checkpoint.metrics['iteration_number']),
        )

        if os.path.exists(checkpoint_dir):
            warnings.warn(f'The contents of the checkpoint {checkpoint_dir} have been overwritten.')

        model_save_dir, optim_file, metrics_file = map(
            lambda x: os.path.join(checkpoint_dir, x),
            [self._model_subdirectory, self._optim_state_filename, self._metrics_filename],
        )

        checkpoint.model.save_pretrained(model_save_dir)
        torch.save(checkpoint.optimizer_state, optim_file)

        with open(metrics_file, 'w') as stream:
            json.dump(checkpoint.metrics, stream, indent=4)
