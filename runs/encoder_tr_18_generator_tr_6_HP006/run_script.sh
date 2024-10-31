/home/sapronov/.virtualenvs/lca-solvers/bin/python3 \
/home/sapronov/lca-solvers/pipeline/__main__.py \
run_name=encoder_tr_18_generator_tr_6_HP006 \
adapter=split_adapter/all_params_6 \
composer=split_composer/python_files_20k_32 \
logger=wandb_logger/wandb_turrets \
preprocessor=split_lm_preprocessor/full_input_loss_8k_16k_2k \
+additional_composer=split_composer/python_files_20k_32 \
+additional_preprocessor=split_lm_preprocessor/full_input_loss_8k_16k