from pipeline.configs.config_base import ConfigBase
from pipeline.outputs.metrics.statistic_base import StatisticName

from dataclasses import dataclass


@dataclass
class CheckpointManagerConfig(ConfigBase):
    main_metric: StatisticName
    directory: str
    
    checkpoint_directory_template: str = '{iteration_number:04d}'
    model_subdirectory: str = 'model'
    optim_state_filename: str = 'optim.pt'
    metrics_filename: str = 'metrics.json'  # should be .json


@dataclass(kw_only=True)
class TopKCheckpointManagerConfig(CheckpointManagerConfig):
    max_checkpoints_num: int
