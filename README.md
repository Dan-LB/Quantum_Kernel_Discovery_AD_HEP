# Quantum_Kernel_Discovery_AD

Work in progress README

# Pacchetti librerie etc:
TO DO


## Download data
Download data from here [https://zenodo.org/records/7673769] and move it into "QML_paper_data" folder

## Kernel Discovery Pipeline
To execute the Quantum Kernel Discovery pipeline for Anomaly Detection:

python main.py mode [--latent_dim VALUE] [--opt_param VALUE] [--n_op VALUE] [--ntest VALUE] [--ntrain VALUE] [--final_train VALUE] [--final_test VALUE]

mode: select "CUSTOM" for the kernel discovery algorithm

--latent_dim VALUE: Specifies the latent dimension
--opt_param VALUE: Specifies the number of Bayesian optimization epochs
--n_op VALUE: Specifies the desired number of operations in the kernel

--ntest VALUE: Specifies the number of samples used to evaluate the model built on the kernel
--ntrain VALUE: Specifies the number of samples used to train the model built on the kernel we want to evaluate

--final_train VALUE: Specifies the number of samples we want to train a model based on the final kernel
--final_test VALUE: Specifies the number of samples used to evaluate the final model.

### Training and testing comparison models

python main.py mode [--latent_dim VALUE] [--final_train VALUE] [--final_test VALUE]

mode: FIXED to use the quantum model proposed in [\cit], CLASSICAL to use a classical OC-SVM.

## Preparing plots and creating graphs

To prepare plots, use:

python prepare_plot_scores.py --quantum_folder [PATH] --comparison_folder [PATH] --mode [MODE] --out_path [PATH] --name_suffix [SUFFIX]

--quantum_folder PATH: Specifies the folder of the trained quantum model with the discovered kernel 
--comparison_folder PATH: Specifies the folder of the trained model that you want to compare

--mode MODE: Specifies the type of compared model. The available options are CLASSICAL and FIXED

--out_path PATH: Specifies the base path to the output file. The actual files created will have the mode and a suffix appended to this path
--name_suffix SUFFIX: Specifies a string that will be appended at the end of the output files named sig_scores_<name_suffix>.npy and bkg_scores<name_suffix>.npy

python prepare_plot_scores.py --quantum_folder trained_qsvms/model_testing_L4_T20 --comparison_folder trained_qsvms/fixed_testing_L4_T20 --mode FIXED --out_path my_plot --name_suffix n40_k5

To create the final plot, use:

python create_plot.py --file_name [NAME] --mode [MODE] --latent_dim [VALUE]

--file_name: Specifies the name of the file generated in the last step
--mode: CLASSICAL or FIXED
--latent_dim: Specifies the latent dimension used

# Example to reproduce our results:

python main.py CUSTOM --latent_dim 8 --final_train 200 --final_test 1500 --n_op 12 --opt_param 10 --ntest 75 --ntrain 75
python main.py FIXED --latent_dim 8 --final_train 200 --final_test 1500

python prepare_plot_scores.py --quantum_folder trained_qsvms/model_testing_L8_T200 --comparison_folder trained_qsvms/fixed_testing_L8_T200 --mode FIXED --out_path plot --name_suffix n1500_k5

python create_plot.py --file_name plot_comparison_fixed --mode FIXED --latent_dim 8
