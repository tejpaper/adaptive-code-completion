python3 -m pipeline \
run_name=file_level \
dataset=file_level \
model=ocoder1p5 \
preprocessor=completion_loss_preprocessor/full_completion_loss_0_4k \
+additional_preprocessor=lm_preprocessor/full_input_loss_12k_4k