python3 -m pipeline \
run_name=half_memory \
dataset=half_memory \
preprocessor=completion_loss_preprocessor/full_completion_loss_12k_4k \
+additional_preprocessor=lm_preprocessor/full_input_loss_12k_4k