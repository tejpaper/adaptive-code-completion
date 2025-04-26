from pipeline.outputs.loggers.logger_base import Log

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class Checkpoint:
    metrics: Log
    model: nn.Module
    optimizer_state: dict
