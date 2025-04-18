from pipeline.data.preprocessors.completion_loss_preprocessor import CompletionLossPreprocessor
from pipeline.data.preprocessors.lm_preprocessor import LMPreprocessor


PREPROCESSORS_REGISTRY = {
    'completion_loss_preprocessor': CompletionLossPreprocessor,
    'lm_preprocessor': LMPreprocessor,
}
