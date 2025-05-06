```bash
python3 -m venv .venv
source .venv/bin/activate

```

```bash
pip install -r requirements/demo.txt
pip install -r requirements/evaluation.txt
pip install -r requirements/incontext.txt
pip install -r requirements/pipeline.txt
```

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation

```

```bash
python3 -m pipeline run_name=test dataset=debugging logger=local_logger/local model=dseek1p3 preprocessor=completion_loss_preprocessor/debugging +additional_preprocessor=completion_loss_preprocessor/debugging split=debugging trainer=universal_trainer/debugging

```
