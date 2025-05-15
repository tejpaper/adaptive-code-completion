<h1 align="center">
  <br>
    Project Adaptation in Code Completion via In-Context Learning
  <br>
</h1>

<h4 align="center">The source codes on bachelor's thesis</h4>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#contributions">Contributions</a> •
  <a href="#structure">Structure</a> •
  <a href="#installation">Installation</a> •
  <a href="#reproduction">Reproduction</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a> •
</p>

## Abstract

This thesis addresses the challenge of enhancing code completion models with repository-level context awareness. Modern completion systems struggle with information dispersed across large codebases, limiting their performance. The work presents a context composition framework that extracts relevant repository information and a fine-tuning pipeline for model adaptation, evaluated through systematic experimentation. The research demonstrates that context selection strategy significantly impacts completion quality during inference, while repository-level pre-training preserves in-context learning capabilities. Notably, the study demonstrates that computational requirements for context window extension can be substantially reduced while maintaining competitive performance, advancing code completion by enabling better integration of project-wide information.

**Keywords:** repository-level code completion, project adaptation, in-context learning, long context, context extension, resource efficiency, Transformer, Code LLM

## Contributions

### Implemented

- **Context Composition Framework** (`incontext`): Modular and flexible package to extract and compose relevant information from software repositories.

- **Fine-Tuning Pipeline** (`pipeline`): End-to-end pipeline for project adaptation of code completion LLMs via fine-tuning.

### Demonstrated

- **Composition Impact on Inference:** Repository context significantly impacts completion quality during inference.

- **Fine-Tuning on Compositions:** [DeepSeek-Coder-Base 1.3B](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) shows minimal effect from context-specific fine-tuning, suggesting rooted initial training of this model.

- **Effect of Context Extension:** Context extension preserves the in-context learning capabilities developed earlier.

- **Influence of Composition on Context Extension:** Repository context plays a minimal role in the outcome of context extension.

- **Resource Efficiency:** ✨ Repository-level pre-training can achieve competitive results with significantly fewer resources (73M tokens vs billions). ✨

- Other, more detailed insights on the subject of the thesis.

### Released

The core checkpoints will be available here soon.

## Structure

```bash
.
├── configs        # configs for pipeline and evaluation
├── datasets       # dataset head demos
├── demo           # demo for incontext package
├── evaluation     # evaluation script and outputs
├── incontext      # context composition framework
├── paper          # LaTeX source files
├── pipeline       # fine-tuning pipeline
├── requirements   # dependency files
├── runs           # experiment instances
└── thesis.pdf     # compiled thesis document
```

## Installation

### Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```
### Dependencies

```bash
pip install -r requirements/demo.txt
pip install -r requirements/evaluation.txt
pip install -r requirements/incontext.txt
pip install -r requirements/pipeline.txt
```

[Flash Attention](https://github.com/Dao-AILab/flash-attention) (optional):

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```
## Reproduction

Running the Pipeline:

```bash
python3 -m pipeline \
    run_name=test \
    dataset=debugging \
    logger=local_logger/local \
    model=dseek1p3 \
    preprocessor=completion_loss_preprocessor/debugging \
    +additional_preprocessor=completion_loss_preprocessor/debugging \
    split=debugging \
    trainer=universal_trainer/debugging
```

The raw training dataset can be recreated using the following file:

```python
import pandas as pd

df = pd.read_parquet('datasets/raw/datapoints.parquet')
print(df.shape)    # (361052, 3)
print(df.columns)  # ['repo', 'commit_hash', 'completion_file']
```

The used benchmark is available [here](https://huggingface.co/datasets/JetBrains-Research/lca-project-level-code-completion).

## License

MIT & LPPL (paper subdirectory)

## Citation

```bibtex
@mastersthesis{sapronov2025projectadaptation,
  author       = {Maksim Sapronov},
  title        = {Project Adaptation in Code Completion via In-Context Learning},
  school       = {Czech Technical University in Prague},
  year         = {2025},
  type         = {Bachelor's thesis},
  address      = {Prague, Czech Republic},
  url          = {https://github.com/tejpaper/project-adaptation-code-completion}
}
```
