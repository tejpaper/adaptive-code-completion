python3 -m pipeline \
run_name=reversed_declarations \
dataset=lazy/reversed_declarations \
preprocessor=completion_loss_preprocessor/full_completion_loss_12k_4k \
+additional_preprocessor=lm_preprocessor/full_input_loss_12k_4k