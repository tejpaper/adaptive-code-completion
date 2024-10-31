from pipeline.data.preprocessors.completion_loss_preprocessor import CompletionLossPreprocessor

import torch


class FileLevelPreprocessor(CompletionLossPreprocessor):
    def calc_lens(self,
                  prompt: list[int],
                  context: list[int],
                  completion: list[int],
                  ) -> tuple[int, int, int]:
        prompt_len, _, completion_len = super().calc_lens(prompt, context, completion)
        return prompt_len, -len(context), completion_len
