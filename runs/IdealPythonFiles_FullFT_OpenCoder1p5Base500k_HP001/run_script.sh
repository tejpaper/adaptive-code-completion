/home/sapronov/.virtualenvs/lca-solvers/bin/python3 \
/home/sapronov/lca-solvers/pipeline/__main__.py \
run_name=IdealPythonFiles_FullFT_OpenCoder1p5Base500k_HP001 \
composer=chained_composer/ideal_python_files \
model=ocoder1p5_theta_500k \
+additional_composer=chained_composer/filled_python_files \
+additional_preprocessor=completion_loss_preprocessor/full_completion_loss_16k