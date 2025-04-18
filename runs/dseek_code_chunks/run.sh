python3 -m pipeline \
run_name=dseek_code_chunks \
dataset=lazy/code_chunks \
model=dseek1p3 \
+additional_preprocessor=lm_preprocessor/full_input_loss_12k_4k