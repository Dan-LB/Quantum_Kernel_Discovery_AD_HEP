python main.py FIXED --latent_dim 4 --final_train 20 --final_test 40

python main.py CUSTOM --latent_dim 4 --final_train 20 --final_test 40 --n_op 3 --opt_param 3 --ntest 5 --ntrain 5


python prepare_plot_scores.py --quantum_folder trained_qsvms/model_testing_L4_T20 --comparison_folder trained_qsvms/fixed_testing_L4_T20 --mode FIXED --out_path my_plot --name_suffix n40_k5



python create_plot.py --file_name my_plot_comparison_fixed --mode FIXED --latent_dim 4
