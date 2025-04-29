python3 -m pipeline \
run_name=debugging \
dataset=debugging \
logger=local_logger/local \
preprocessor=completion_loss_preprocessor/debugging \
+additional_preprocessor=completion_loss_preprocessor/debugging \
split=debugging \
trainer=universal_trainer/debugging