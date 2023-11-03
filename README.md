# Quantum Kernel Discovery for Anomaly Detection

## Introduction
This repository contains the implementation of a Quantum Kernel Discovery pipeline for Anomaly Detection.

## Environment Setup
Before running the pipeline, ensure that the environment is correctly set up by installing the required packages.

### Using pip
Install the dependencies from the `requirements.txt` file using pip:

```
pip install -r requirements.txt
```

### Using conda
Alternatively, create a conda environment using the `environment.yml` file:

```
conda env create -f environment.yml
```

## Data Acquisition
The dataset can be downloaded from the following link: [Dataset](https://zenodo.org/records/7673769). After downloading, move the data into the `QML_paper_data` folder for further processing.

## Quantum Kernel Discovery Pipeline
Execute the pipeline using the command below. The `CUSTOM` mode activates the kernel discovery algorithm.

```
python main.py mode [--latent_dim VALUE] [--opt_param VALUE] [--n_op VALUE] [--ntest VALUE] [--ntrain VALUE] [--final_train VALUE] [--final_test VALUE]
```

Arguments:
- `--latent_dim VALUE`: Set the latent dimension.
- `--opt_param VALUE`: Define the number of Bayesian optimization epochs.
- `--n_op VALUE`: Determine the number of operations in the kernel.
- `--ntest VALUE`: Specify the number of samples for model evaluation.
- `--ntrain VALUE`: Set the number of samples for training the model.
- `--final_train VALUE`: Indicate the training sample size for the final model.
- `--final_test VALUE`: Set the evaluation sample size for the final model.

## Model Comparison
To compare the performance of different models:

```
python main.py mode [--latent_dim VALUE] [--final_train VALUE] [--final_test VALUE]
```

Modes:
- `FIXED`: Use the predefined quantum model proposed in [paper di woz et al.].
- `CLASSICAL`: Use a classical OC-SVM.

## Visualization
For generating plots and graphs, follow these instructions:

### Preparing Plots
```
python prepare_plot_scores.py --quantum_folder [PATH] --comparison_folder [PATH] --mode [MODE] --out_path [PATH] --name_suffix [SUFFIX]
```

Parameters:
- `--quantum_folder PATH`: Path to the quantum model's folder.
- `--comparison_folder PATH`: Path to the comparison model's folder.
- `--mode MODE`: Model type (`CLASSICAL` or `FIXED`).
- `--out_path PATH`: Base path for the output files.
- `--name_suffix SUFFIX`: Suffix for the output filenames.

### Creating Final Plots
```
python create_plot.py --file_name [NAME] --mode [MODE] --latent_dim [VALUE]
```

Arguments:
- `--file_name NAME`: Name of the file from the previous step.
- `--mode MODE`: Type of model (`CLASSICAL` or `FIXED`).
- `--latent_dim VALUE`: The latent dimension used.

## Reproducible Results
To reproduce our findings, use the following commands:

```
python main.py CUSTOM --latent_dim 8 --final_train 200 --final_test 1500 --n_op 12 --opt_param 10 --ntest 75 --ntrain 75
python main.py FIXED --latent_dim 8 --final_train 200 --final_test 1500

python prepare_plot_scores.py --quantum_folder trained_qsvms/model_testing_L8_T200 --comparison_folder trained_qsvms/fixed_testing_L8_T200 --mode FIXED --out_path plot --name_suffix n1500_k5

python create_plot.py --file_name plot_comparison_fixed --mode FIXED --latent_dim 8
```

