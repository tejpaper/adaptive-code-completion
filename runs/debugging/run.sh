python3 -m pipeline \
run_name=debugging \
dataset=debugging \
logger=local_logger/local \
+additional_preprocessor=completion_loss_preprocessor/debugging \
trainer=universal_trainer/debugging \
split=debugging