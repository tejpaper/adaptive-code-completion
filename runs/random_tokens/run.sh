python3 -m pipeline \
run_name=random_tokens \
dataset=lazy/random_tokens \
preprocessor=completion_loss_preprocessor/full_completion_loss_12k_4k \
+additional_preprocessor=lm_preprocessor/full_input_loss_12k_4k