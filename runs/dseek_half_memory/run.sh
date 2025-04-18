python3 -m pipeline \
run_name=dseek_half_memory \
dataset=lazy/half_memory \
model=dseek1p3 \
preprocessor=completion_loss_preprocessor/full_completion_loss_12k_4k \
+additional_preprocessor=lm_preprocessor/full_input_loss_12k_4k