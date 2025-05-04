import random
from collections import defaultdict

import pandas as pd
from datasets import Dataset


def train_test_split(df: pd.DataFrame,
                     test_size: int,
                     upper_bound_per_repo: int,
                     random_seed: int | None = None,
                     ) -> tuple[list[int], list[int] | None]:
    if test_size == 0:
        return list(range(len(df))), None

    generator = random.Random(random_seed)
    queue = defaultdict(list)
    repos_enum = list(enumerate(df.pre_context_prompt))
    generator.shuffle(repos_enum)

    for idx, repo in repos_enum:
        queue[repo].append(idx)

    queue = list(queue.items())
    generator.shuffle(queue)

    train_repos_ids = set(range(len(df)))
    test_repos_ids = set()
    cur_test_size = 0

    while cur_test_size != test_size:
        if queue:
            repo, ids = queue.pop()
        else:
            raise ValueError(
                'There are not enough data points in the original dataset to satisfy both the '
                'test_size and upper_bound_per_repo arguments. Try either decreasing the test_size '
                'or increasing the upper_bound_per_repo.')

        num_new_samples = min(upper_bound_per_repo, test_size - cur_test_size, len(ids))

        train_repos_ids.difference_update(ids)
        test_repos_ids.update(ids[:num_new_samples])
        cur_test_size += num_new_samples

    return list(train_repos_ids), list(test_repos_ids)


def reverse_context(string: str) -> str:
    # Experiments with reverse context are only done with OpenCoder, so this function is sufficient
    # See the dual class ReverseContextPostprocessor in incontext/blocks/context_postprocessing.py
    return '<file_sep>'.join(string.split('<file_sep>')[:-1][::-1] + [''])


def load_dataset(main_dataset_path: str,
                 add_dataset_path: str | None,
                 reversed_context: bool,
                 file_level: bool,
                 **split_kwargs,
                 ) -> tuple[Dataset, Dataset, Dataset | None]:
    dataset = pd.read_parquet(main_dataset_path)
    train_ids, test_ids = train_test_split(dataset, **split_kwargs)

    if reversed_context:
        dataset['composed_context'] = dataset.composed_context.apply(reverse_context)
    if file_level:
        dataset['composed_context'] = ''

    train_ds = Dataset.from_pandas(dataset.iloc[train_ids])
    valid_ds = Dataset.from_pandas(dataset.iloc[test_ids])

    if add_dataset_path is None:
        return train_ds, valid_ds, None

    add_dataset = pd.read_parquet(add_dataset_path)
    assert list(dataset.pre_context_prompt) == list(add_dataset.pre_context_prompt)
    add_valid_ds = Dataset.from_pandas(add_dataset.iloc[test_ids])

    return train_ds, valid_ds, add_valid_ds
