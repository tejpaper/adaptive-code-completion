from incontext.init_from_config import find_class
from pipeline.outputs.metrics.counters import EpochCounter, TokenCounter
from pipeline.outputs.metrics.cross_entropy import CrossEntropy
from pipeline.outputs.metrics.exact_match import ExactMatch
from pipeline.outputs.metrics.statistic_base import LazyStatistic, StatisticName, StatisticBase
from pipeline.outputs.metrics.top_k_accuracy import TopKAccuracy

import os
from typing import Iterable

import yaml
from transformers import PreTrainedTokenizerBase


def find_metric_class(metric_name: str) -> type:
    return find_class(
        name=metric_name,
        module_name='pipeline.outputs.metrics',
        normalization_func=lambda x: {
            'epoch': 'EpochCounter',
            'num_tokens': 'TokenCounter',
        }.get(x, x.replace('_', '')),
    )


def init_metrics(loaded_config: Iterable[StatisticName],
                 configs_dir: str,
                 tokenizer: PreTrainedTokenizerBase,
                 ) -> list[StatisticBase]:
    # Dictionary solves the problem of using the same metric names multiple times:
    # Only the last metric specified in the configuration will be used
    metrics = dict()

    for path in loaded_config:
        full_path = os.path.join(configs_dir, 'metrics/metrics', path)
        metric_name = os.path.basename(os.path.dirname(path))

        with open(full_path) as stream:
            metric_config = yaml.safe_load(stream)

        if metric_config is None:
            metric_config = dict()

        metric_class = find_metric_class(metric_name)
        if metric_class.requires_tokenizer:
            metric_config['tokenizer'] = tokenizer

        metric = metric_class(**metric_config)
        metrics[metric.name] = metric

    return list(metrics.values())


__all__ = [
    'find_metric_class',
    'init_metrics',
    'EpochCounter',
    'TokenCounter',
    'CrossEntropy',
    'ExactMatch',
    'LazyStatistic',
    'TopKAccuracy',
]
