from pipeline.outputs.metrics.metric_base import OptimizationMode, MetricBase, MaskType, MaskBasedMetric
from pipeline.outputs.metrics.statistic_base import StatisticValue, StatisticName

from typing import TypeVar, Type

import torch

T = TypeVar('T')

# avoids cyclical imports
UniversalTrainer = TypeVar('UniversalTrainer')


class EpochCounter(MetricBase):
    _instance = None  # singleton pattern
    # It is not a metric in the usual sense, but when used with main_metric in
    # TopKCheckpointManager, it will result in saving only the last k checkpoints
    mode = OptimizationMode.MAX

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.samples = 0
        self.ds_len = 1

    @property
    def name(self) -> StatisticName:
        return 'epoch'

    def micro_batch_update(self, input_ids: torch.Tensor, trainer: UniversalTrainer, **_kwargs) -> None:
        if trainer.model.training:  # ignores validation samples
            self.samples += input_ids.shape[0]
            self.ds_len = len(trainer.train_dl.dataset)

    def batch_commit(self, **_kwargs) -> StatisticValue:
        return self.samples / self.ds_len


class TokenCounter(MaskBasedMetric):
    mode = OptimizationMode.MAX

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_tokens = 0

    @property
    def name(self) -> StatisticName:
        mask_type = '' if self.mask_type == MaskType.ATTACHED else f'{self.mask_type}_'
        return f'num_{mask_type}tokens'

    @torch.inference_mode()
    def micro_batch_update(self, **kwargs) -> None:
        self.num_tokens += self.get_mask(**kwargs).sum().item()

    def batch_commit(self, **_kwargs) -> StatisticValue:
        return self.num_tokens
