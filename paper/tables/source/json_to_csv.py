import json
import os

import pandas as pd

PATH_TO_EVALUATION_OUTPUTS = '../../../evaluation/outputs'


def compile_rq_a1_table(path_to_results: str, output_file: str = 'csv/rq_a1.csv') -> None:
    output_file = os.path.join(PATH_TO_EVALUATION_OUTPUTS, output_file)
    results = list()

    for filename in os.listdir(path_to_results):
        if 'rq_a1' not in filename:
            continue

        path_to_json = os.path.join(path_to_results, filename)
        with open(path_to_json) as stream:
            result = json.load(stream)

        results.append(dict(
            eval_name=filename.replace('.json', '').replace('rq_a1_', ''),
            is_contaminated=result['allow_leak'],
            inproject_em=result['exact_match']['inproject']['value'],
            infile_em=result['exact_match']['infile']['value'],
        ))

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def compile_rq_a2_table(path_to_results: str, output_file: str = 'csv/rq_a2.csv') -> None:
    output_file = os.path.join(PATH_TO_EVALUATION_OUTPUTS, output_file)
    results = list()

    eval_names = [
        'code_chunks_2',
        'declarations',
        'docstring_and_comment_chunks',
        'filled_python_files_2',
        'filled_random_files',
        'filled_text_files',
        'half_memory',
        'iou_python_files_2',
        'random_python_files_2',
        'completion_duplication_2',
        'completion_leak',
        'file_level',
        'random_tokens',
    ]

    for composer_type in ('fl_16k', 'pd_16k', 'or_16k'):
        for eval_name in eval_names:
            path_to_json = os.path.join(path_to_results, f'rq_a2_{composer_type}_{eval_name}.json')

            with open(path_to_json) as stream:
                result = json.load(stream)

            results.append(dict(
                eval_name=eval_name,
                composer_type=composer_type,
                is_contaminated=result['allow_leak'],
                inproject_em=result['exact_match']['inproject']['value'],
                infile_em=result['exact_match']['infile']['value'],
            ))

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def compile_rq_b_table(path_to_results: str, output_file: str = 'csv/rq_b.csv') -> None:
    output_file = os.path.join(PATH_TO_EVALUATION_OUTPUTS, output_file)
    results = list()

    eval_names = [
        'no_extension',
        'no_extension_2',
        'code_chunks_2',
        'declarations',
        'docstring_and_comment_chunks',
        'filled_python_files_2',
        'filled_random_files',
        'filled_text_files',
        'half_memory',
        'iou_python_files_2',
        'irrelevant_code_chunks_2',
        'irrelevant_declarations',
        'irrelevant_docstring_and_comment_chunks',
        'irrelevant_filled_python_files_2',
        'irrelevant_filled_text_files',
        'irrelevant_half_memory',
        'irrelevant_iou_python_files_2',
        'random_python_files_2',
        'reversed_code_chunks_2',
        'reversed_declarations',
        'reversed_docstring_and_comment_chunks',
        'reversed_filled_python_files_2',
        'reversed_filled_text_files',
        'reversed_half_memory',
        'reversed_iou_python_files_2',
        'completion_duplication_2',
        'completion_leak',
        'file_level',
        'file_level_2',
        'irrelevant_completion_leak',
        'random_tokens',
        'reversed_completion_leak',
    ]

    for composer_type in ('fl_4k', 'pd_4k', 'pd_16k', 'or_16k'):
        for eval_name in eval_names:
            if 'no_extension' in eval_name and composer_type == 'or_16k':
                continue

            path_to_json = os.path.join(path_to_results, f'rq_b_{composer_type}_{eval_name}.json')

            with open(path_to_json) as stream:
                result = json.load(stream)

            results.append(dict(
                eval_name=eval_name,
                composer_type=composer_type,
                is_contaminated=result['allow_leak'],
                inproject_em=result['exact_match']['inproject']['value'],
                infile_em=result['exact_match']['infile']['value'],
            ))

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def compile_rq_a2_gradient_masking_table(
        path_to_results: str,
        output_file: str = 'csv/rq_a2_gradient_masking.csv') -> None:
    output_file = os.path.join(PATH_TO_EVALUATION_OUTPUTS, output_file)
    results = list()

    eval_names = [
        'code_chunks', 'code_chunks_2',
        'filled_python_files', 'filled_python_files_2',
        'iou_python_files', 'iou_python_files_2',
        'random_python_files', 'random_python_files_2',
        'completion_duplication', 'completion_duplication_2',
    ]

    for composer_type in ('fl_16k', 'pd_16k', 'or_16k'):
        for eval_name in eval_names:
            path_to_json = os.path.join(path_to_results, f'rq_a2_{composer_type}_{eval_name}.json')

            with open(path_to_json) as stream:
                result = json.load(stream)

            results.append(dict(
                eval_name=eval_name,
                composer_type=composer_type,
                is_masked=eval_name.endswith('_2'),
                is_contaminated=result['allow_leak'],
                inproject_em=result['exact_match']['inproject']['value'],
                infile_em=result['exact_match']['infile']['value'],
            ))

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def compile_rq_b_gradient_masking_table(
        path_to_results: str,
        output_file: str = 'csv/rq_b_gradient_masking.csv') -> None:
    output_file = os.path.join(PATH_TO_EVALUATION_OUTPUTS, output_file)
    results = list()

    eval_names = [
        'code_chunks', 'code_chunks_2',
        'filled_python_files', 'filled_python_files_2',
        'iou_python_files', 'iou_python_files_2',
        'irrelevant_code_chunks', 'irrelevant_code_chunks_2',
        'irrelevant_filled_python_files', 'irrelevant_filled_python_files_2',
        'irrelevant_iou_python_files', 'irrelevant_iou_python_files_2',
        'random_python_files', 'random_python_files_2',
        'reversed_code_chunks', 'reversed_code_chunks_2',
        'reversed_filled_python_files', 'reversed_filled_python_files_2',
        'reversed_iou_python_files', 'reversed_iou_python_files_2',
        'completion_duplication', 'completion_duplication_2',
    ]

    for composer_type in ('fl_4k', 'pd_4k', 'pd_16k', 'or_16k'):
        for eval_name in eval_names:
            path_to_json = os.path.join(path_to_results, f'rq_b_{composer_type}_{eval_name}.json')

            with open(path_to_json) as stream:
                result = json.load(stream)

            results.append(dict(
                eval_name=eval_name,
                composer_type=composer_type,
                is_masked=eval_name.endswith('_2'),
                is_contaminated=result['allow_leak'],
                inproject_em=result['exact_match']['inproject']['value'],
                infile_em=result['exact_match']['infile']['value'],
            ))

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def compile_beyond_training_window_table(
        path_to_results: str,
        output_file: str = 'csv/beyond_training_window.csv') -> None:
    output_file = os.path.join(PATH_TO_EVALUATION_OUTPUTS, output_file)
    results = list()

    eval_names = [
        'rq_b_pd_1k_file_level_2',
        'rq_b_pd_2k_file_level_2',
        'rq_b_pd_4k_file_level_2',
        'rq_b_pd_8k_file_level_2',
        'rq_b_pd_16k_file_level_2',
        'rq_b_pd_32k_file_level_2',
        'rq_b_pd_64k_file_level_2',
        'rq_b_pd_128k_file_level_2',
        'rq_b_pd_1k_filled_python_files_2',
        'rq_b_pd_2k_filled_python_files_2',
        'rq_b_pd_4k_filled_python_files_2',
        'rq_b_pd_8k_filled_python_files_2',
        'rq_b_pd_16k_filled_python_files_2',
        'rq_b_pd_32k_filled_python_files_2',
        'rq_b_pd_64k_filled_python_files_2',
        'rq_b_pd_128k_filled_python_files_2',
        'rq_b_pd_1k_qwen',
        'rq_b_pd_2k_qwen',
        'rq_b_pd_4k_qwen',
        'rq_b_pd_8k_qwen',
        'rq_b_pd_16k_qwen',
        'rq_b_pd_32k_qwen',
        'rq_b_pd_64k_qwen',
        'rq_b_pd_128k_qwen',
        'rq_b_pd_1k_dseek',
        'rq_b_pd_2k_dseek',
        'rq_b_pd_4k_dseek',
        'rq_b_pd_8k_dseek',
        'rq_b_pd_16k_dseek',
        'rq_b_pd_32k_dseek',
        'rq_b_pd_64k_dseek',
        'rq_b_pd_128k_dseek',
    ]

    for eval_name in eval_names:
        path_to_json = os.path.join(path_to_results, f'{eval_name}.json')

        with open(path_to_json) as stream:
            result = json.load(stream)

        model_arg = result['sh'].split('\\\n')[-1]
        if 'model=' not in model_arg:
            model = 'ocoder1p5_theta_500k'
        else:
            model = model_arg[6:]

        results.append(dict(
            eval_name=eval_name,
            model=model,
            inproject_em=result['exact_match']['inproject']['value'],
            infile_em=result['exact_match']['infile']['value'],
        ))

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def main() -> None:
    path_to_results = os.path.join(PATH_TO_EVALUATION_OUTPUTS, 'json')
    compile_rq_a1_table(path_to_results)
    compile_rq_a2_table(path_to_results)
    compile_rq_b_table(path_to_results)

    compile_rq_a2_gradient_masking_table(path_to_results)
    compile_rq_b_gradient_masking_table(path_to_results)

    compile_beyond_training_window_table(path_to_results)


if __name__ == '__main__':
    main()
