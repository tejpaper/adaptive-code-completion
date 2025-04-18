from pipeline.configs.adapter_config import (
    AdapterConfig,
)
from pipeline.configs.checkpointer_config import (
    CheckpointManagerConfig,
    TopKCheckpointManagerConfig,
)
from pipeline.configs.composer_config import (
    ChainedComposerConfig,
)
from pipeline.configs.config_base import ConfigBase
from pipeline.configs.logger_config import (
    LocalLoggerConfig,
    WandbLoggerConfig,
)
from pipeline.configs.preprocessor_config import (
    PreprocessorConfig,
)
from pipeline.configs.trainer_config import UniversalTrainerConfig

CONFIGS_REGISTRY = {
    # adapters
    'identity_adapter': AdapterConfig,

    # checkpointers
    'checkpointer': CheckpointManagerConfig,
    'top_k_checkpointer': TopKCheckpointManagerConfig,

    # loggers
    'dummy_logger': ConfigBase,
    'local_logger': LocalLoggerConfig,
    'wandb_logger': WandbLoggerConfig,

    # composers
    'chained_composer': ChainedComposerConfig,

    # preprocessors
    'completion_loss_preprocessor': PreprocessorConfig,
    'lm_preprocessor': PreprocessorConfig,

    # trainers
    'universal_trainer': UniversalTrainerConfig,
}
