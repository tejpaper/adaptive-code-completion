################### init.py ###################

from pipeline.configs.configs_registry import CONFIGS_REGISTRY
from pipeline.data.composers.blocks.blocks_registry import BLOCKS_REGISTRY
from pipeline.data.composers.composer_base import ComposerBase
from pipeline.data.composers.composers_registry import COMPOSERS_REGISTRY

import os

import yaml
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase


def init_composer(
    cls_name: str,
    loaded_config: DictConfig,
    configs_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    **kwargs,
) -> ComposerBase:
    config = CONFIGS_REGISTRY[cls_name].from_dict(dict(loaded_config) | kwargs)

    if cls_name == 'chained_composer':
        for path in loaded_config.block_configs:
            full_path = os.path.join(
                configs_dir, 'composer/chained_composer/blocks', path
            )
            block_name = os.path.basename(os.path.dirname(path))

            with open(full_path) as stream:
                block_config = yaml.safe_load(stream)

            if block_config is None:
                block_config = dict()

            block_cls = BLOCKS_REGISTRY[block_name]
            if block_cls.requires_tokenizer:
                block_config['tokenizer'] = tokenizer

            block = block_cls(**block_config)
            config.blocks.append(block)

    composer = COMPOSERS_REGISTRY[cls_name](**config.dict)
    return composer


################### composers_registry.py ###################

from pipeline.data.composers.chained_composer import ChainedComposer

COMPOSERS_REGISTRY = {
    'chained_composer': ChainedComposer,
}

################### blocks_registry.py ###################

from pipeline.data.composers.blocks.chunk_harvesting import (
    JoiningHarvester,
    PathCommentHarvester,
)
from pipeline.data.composers.blocks.chunk_ranking import (
    NegativePathDistanceRanker,
    FunctionCallRanker,
    FileExtensionRanker,
    RandomRanker,
    IoURanker,
    IdealRanker,
)
from pipeline.data.composers.blocks.chunk_sorting import (
    LexicographicSorter,
)
from pipeline.data.composers.blocks.context_postprocessing import (
    PartialMemoryPostprocessor,
    LineLengthPostprocessor,
    LineStripPostprocessor,
    InverseFrequencyMemoryPostprocessor,
)
from pipeline.data.composers.blocks.file_chunking import (
    FileGrainedChunker,
    CodeSegmentGrainedChunker,
    FixedLineChunker,
)
from pipeline.data.composers.blocks.file_filtering import (
    NullFileFilter,
    InclusiveFileExtensionFilter,
    ExclusiveFileExtensionFilter,
    EmptyFileFilter,
    FileLengthFilter,
    TokenizedFileLengthFilter,
    CharTokenRatioFilter,
    IndexQuantileGroupFilter,
)
from pipeline.data.composers.blocks.file_preprocessing import (
    EmptyLinesRemovalPreprocessor,
    NewlinePreprocessor,
    DeclarationOnlyPreprocessor,
)

BLOCKS_REGISTRY = {
    # file_filtering
    'null_file_filter': NullFileFilter,
    'inclusive_file_extension_filter': InclusiveFileExtensionFilter,
    'exclusive_file_extension_filter': ExclusiveFileExtensionFilter,
    'empty_file_filter': EmptyFileFilter,
    'file_length_filter': FileLengthFilter,
    'tokenized_file_length_filter': TokenizedFileLengthFilter,
    'char_token_ratio_filter': CharTokenRatioFilter,
    'index_quantile_group_filter': IndexQuantileGroupFilter,
    # file_preprocessing
    'empty_lines_removal_preprocessor': EmptyLinesRemovalPreprocessor,
    'newline_preprocessor': NewlinePreprocessor,
    'declaration_only_preprocessor': DeclarationOnlyPreprocessor,
    # file_chunking
    'file_grained_chunker': FileGrainedChunker,
    'code_segment_grained_chunker': CodeSegmentGrainedChunker,
    'fixed_line_chunker': FixedLineChunker,
    # chunk_ranking
    'negative_path_distance_ranker': NegativePathDistanceRanker,
    'function_call_ranker': FunctionCallRanker,
    'file_extension_ranker': FileExtensionRanker,
    'random_ranker': RandomRanker,
    'iou_ranker': IoURanker,
    'ideal_ranker': IdealRanker,
    # chunk_sorting
    'lexicographic_sorter': LexicographicSorter,
    # chunk_harvesting
    'joining_harvester': JoiningHarvester,
    'path_comment_harvester': PathCommentHarvester,
    # context_postprocessing
    'partial_memory_postprocessor': PartialMemoryPostprocessor,
    'line_length_postprocessor': LineLengthPostprocessor,
    'line_strip_postprocessor': LineStripPostprocessor,
    'inverse_frequency_memory_postprocessor': InverseFrequencyMemoryPostprocessor,
}

################### composer_config.py ###################

from pipeline.configs.config_base import ConfigBase
from pipeline.data.composers.chain import ComposerBlock

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ChainedComposerConfig(ConfigBase):
    pre_context_prompt: str
    post_context_prompt: str
    path_comment_template: str
    blocks: Sequence[ComposerBlock] = field(default_factory=list)


def main() -> None:
    pass  # TODO: script for dataset composition


if __name__ == '__main__':
    main()
