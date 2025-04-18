from pipeline.configs.config_base import ConfigBase

from dataclasses import dataclass


@dataclass
class SplitConfig(ConfigBase):  # TODO: determine if this is needed
    test_size: int  # 0 means no validation at all
    upper_bound_per_repo: int
    random_seed: int | None
