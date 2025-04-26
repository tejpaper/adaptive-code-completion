from pipeline.outputs.metrics.counters import EpochCounter, TokenCounter
from pipeline.outputs.metrics.cross_entropy import CrossEntropy
from pipeline.outputs.metrics.exact_match import ExactMatch
from pipeline.outputs.metrics.statistic_base import LazyStatistic
from pipeline.outputs.metrics.top_k_accuracy import TopKAccuracy

METRICS_REGISTRY = {
    # metrics
    'cross_entropy': CrossEntropy,
    'exact_match': ExactMatch,  # TODO: handle this confusion with two names
    'completion_exact_match': ExactMatch,
    'top_k_accuracy': TopKAccuracy,

    # statistics
    'epoch': EpochCounter,
    'num_tokens': TokenCounter,
    'lazy_statistic': LazyStatistic,
}
