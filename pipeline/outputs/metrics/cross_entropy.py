from pipeline.outputs.metrics.metric_base import OptimizationMode, MaskBasedMetric
from pipeline.outputs.metrics.statistic_base import StatisticValue

import torch


class CrossEntropy(MaskBasedMetric):
    mode = OptimizationMode.MIN

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean_loss = 0
        self.num_tokens = 0

    @torch.inference_mode()
    def micro_batch_update(self, loss_per_token: torch.Tensor, **kwargs) -> None:
        mask = self.get_mask(**kwargs)
        loss_update = torch.nan_to_num(loss_per_token[mask].mean()).item()
        num_tokens_update = mask.sum().item()

        # loss correction w.r.t. number of masked tokens (for unbalanced batches)
        if not self.num_tokens:
            self.mean_loss += loss_update
            self.num_tokens = 0
        else:
            tokens_ratio = num_tokens_update / self.num_tokens
            self.mean_loss += tokens_ratio * loss_update
            self.mean_loss /= tokens_ratio + 1

        self.num_tokens += num_tokens_update

    def batch_commit(self, **_kwargs) -> StatisticValue:
        batch_metric = float('nan') if not self.num_tokens else self.mean_loss
        self.mean_loss = 0
        self.num_tokens = 0
        return batch_metric
