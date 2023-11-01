# The quantum and classical kernel machine training script.
# The model is instantiated with some parameters, the data encoding circuit is built,
# it is trained on a data set, and is saved in a folder.

import argparse
import json
from time import perf_counter
from typing import Callable
from qiskit.utils import algorithm_globals
import os
import copy

import pickle


import numpy as np

import qad.algorithms.kernel_machines.util as util
import qad.algorithms.kernel_machines.data_processing as data_processing
from qad.algorithms.kernel_machines.terminal_enhancer import tcols

from qiskit_machine_learning.kernels import FidelityQuantumKernel

seed = 12345
algorithm_globals.random_seed = seed

#params_def:

CUSTOM = "CUSTOM"
FIXED = "FIXED"
CLASSIC = "CLASSIC"


unsup = True
feature_map = "u_dense_encoding"
run_type = "ideal"
output_folder = "quantum_test"

def get_problems_signature_and_paths(L):
    basic_path = "QML_paper_data/latent"+str(L)+"/"
    bkg_path = "QML_paper_data\latent"+str(L)+"\latentrep_QCD_sig.h5"
    test_bkg_path = "QML_paper_data/latent"+str(L)+"/latentrep_QCD_sig_testclustering.h5"

    p1 = basic_path+"latentrep_AtoHZ_to_ZZZ_35.h5"
    p2 = basic_path+"latentrep_RSGraviton_WW_BR_15.h5"
    p3 = basic_path+"latentrep_RSGraviton_WW_NA_35.h5"

    problems_signature = [p1, p2, p3]

    return problems_signature, bkg_path, test_bkg_path

def to_optimize_kernel(kernel, ntrain, ntest, L, problem_id):
    n_qubits = L
    problems_signature, bkg_path, test_bkg_path = get_problems_signature_and_paths(L)
    args = {
        "sig_path": problems_signature[problem_id],
        "bkg_path": bkg_path,
        "test_bkg_path": test_bkg_path,
        "c_param": 1.0,
        "nu_param": 0.01,
        "output_folder": output_folder,
        "gamma": "scale",
        "quantum": True,
        "unsup": True,
        "nqubits": n_qubits,
        "feature_map": "None", #lol
        "backend_name": "statevector_simulator",
        "ibmq_api_config": None,
        "run_type": "ideal",
        "ntrain": ntrain,
        "ntest": ntest,
        "seed": seed,  # For the data shuffling.
    }
    config_ideal = {"seed_simulator": seed}

    switcher = {
        "ideal": lambda: config_ideal,
    }
    args["config"] = switcher.get("ideal", lambda: None)()

    train_loader, test_loader = data_processing.get_data(args, print_info=False)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]

    model = util.init_kernel_machine(args, kernel=kernel, use_custom_kernel=True)

    time_and_train(model.fit, train_features, train_labels)

    out_path = "trained_qsvms/model_to_optimize_kernel"

    test_acc_value = util.eval_metrics(
            model, train_features, train_labels, test_features, test_labels, out_path
    )
    return test_acc_value

def save_kernel(kernel, name):
    """
    It saves a pennylane kernel into a qiskit kernel
    
    """
    out_path = "trained_qsvms/"+name
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    out_path += "/kernel.pkl"

    karnel_qiskit = kernel.to_qiskit_circuit()

    with open(out_path, 'wb') as file:
        pickle.dump(karnel_qiskit, file)

    return 0

def load_kernel(name):
    """
    It loads a qiskit kernel
    """

    out_path = "trained_qsvms/"+name+ "/kernel.pkl"
    with open(out_path, 'rb') as file:
        loaded_quantum_kernel = pickle.load(file)
    
    return loaded_quantum_kernel

def create_model_from_qiskit_circuit(kernel, L):

    """
    kernel should be a qiskit circuit. #change name of func maybe
    returns the model and args
    """
    problems_signature, bkg_path, test_bkg_path = get_problems_signature_and_paths(L)
    n_qubits = L
    args = {
        "sig_path": None,
        "bkg_path": bkg_path,
        "test_bkg_path": test_bkg_path,
        "c_param": 1.0,
        "nu_param": 0.01,
        "output_folder": output_folder,
        "gamma": "scale",
        "quantum": True,
        "unsup": True,
        "nqubits": n_qubits,
        "feature_map": "None", #lol
        "backend_name": "statevector_simulator",
        "ibmq_api_config": None,
        "run_type": "ideal",
        "ntrain": 0,
        "ntest": 0,
        "seed": seed,  # For the data shuffling.
    }
    config_ideal = {"seed_simulator": seed}

    switcher = {
        "ideal": lambda: config_ideal,
    }
    args["config"] = switcher.get("ideal", lambda: None)()

    #circuit = kernel.to_qiskit_circuit()
    kernel = FidelityQuantumKernel(feature_map=kernel)

    model = util.init_kernel_machine(args, kernel=kernel, use_custom_kernel=True)
    return model, args

def create_model_with_defaul_kernel(L):
    problems_signature, bkg_path, test_bkg_path = get_problems_signature_and_paths(L)
    n_qubits = L
    args = {
        "sig_path": None,
        "bkg_path": bkg_path,
        "test_bkg_path": test_bkg_path,
        "c_param": 1.0,
        "nu_param": 0.01,
        "output_folder": output_folder,
        "gamma": "scale",
        "quantum": True,
        "unsup": True,
        "nqubits": n_qubits,
        "feature_map": "u_dense_encoding",
        "backend_name": "statevector_simulator",
        "ibmq_api_config": None,
        "run_type": "ideal",
        "ntrain": 0,
        "ntest": 0,
        "seed": seed,  # For the data shuffling.
        "kfolds":5,
    }
    config_ideal = {"seed_simulator": seed}

    switcher = {
        "ideal": lambda: config_ideal,
    }
    args["config"] = switcher.get("ideal", lambda: None)()


    model = util.init_kernel_machine(args, use_custom_kernel=True)
    return model, args


def create_model_classic(L):
    problems_signature, bkg_path, test_bkg_path = get_problems_signature_and_paths(L)
    n_qubits = L
    args = {
        "sig_path": None,
        "bkg_path": bkg_path,
        "test_bkg_path": test_bkg_path,
        "c_param": 1.0,
        "nu_param": 0.01,
        "output_folder": output_folder,
        "gamma": "scale",
        "quantum": False,
        "unsup": True,
        "nqubits": n_qubits,
        "feature_map": "u_dense_encoding",
        "backend_name": "statevector_simulator",
        "ibmq_api_config": None,
        "run_type": "ideal",
        "ntrain": 0,
        "ntest": 0,
        "seed": seed,  # For the data shuffling.
        "kfolds":5,
    }
    config_ideal = {"seed_simulator": seed}

    switcher = {
        "ideal": lambda: config_ideal,
    }
    args["config"] = switcher.get("ideal", lambda: None)()


    model = util.init_kernel_machine(args, use_custom_kernel=False)
    return model, args

def load_trained_model(name):

    path = "trained_qsvms/"+name+"/model"
    model = util.load_model(path)

    return model

def train_model(model, args, ntrain, ntest, L, problem_id, mode = CUSTOM):
    """
    It should train a model on a certain problem id and save everything
    somewhere.
    """
    problems_signature, bkg_path, test_bkg_path = get_problems_signature_and_paths(L)

    args["sig_path"] = problems_signature[problem_id]
    args["ntrain"] = ntrain
    args["ntest"] = ntest

    train_loader, test_loader = data_processing.get_data(args, print_info=False)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]

    print("Training model on "+str(ntrain))

    time_and_train(model.fit, train_features, train_labels)

    if mode == CUSTOM:
        out_path = "trained_qsvms/model_testing"
    if mode == FIXED:
        out_path = "trained_qsvms/fixed_testing"
    if mode == CLASSIC:
        out_path = "trained_qsvms/classic_testing"

    out_path += "_L"+str(L)+"_T"+str(ntrain)+"_P"+str(problem_id)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    util.print_model_info(model)
    util.export_hyperparameters(model, out_path) #questo non fa niente
    util.save_model(model, out_path)

    test_acc_value = util.eval_metrics(
            model, train_features, train_labels, test_features, test_labels, out_path)

    return 0

def test_model(model, ntest, L, problem_id, n_train = 0, mode = CUSTOM):

    problems_signature, bkg_path, test_bkg_path = get_problems_signature_and_paths(L)
    args = {
        "sig_path": problems_signature[problem_id],
        "bkg_path": bkg_path,
        "test_bkg_path": test_bkg_path,
        "c_param": 1.0,
        "nu_param": 0.01,
        "output_folder": output_folder,
        "gamma": "scale",
        "quantum": True,
        "unsup": True,
        "nqubits": L, #!!!!
        "feature_map": "None", #lol
        "backend_name": "statevector_simulator",
        "ibmq_api_config": None,
        "run_type": "ideal",
        "ntrain": 0,
        "ntest": ntest,
        "seed": seed,  # For the data shuffling.
        "kfolds": 5
    }
    config_ideal = {"seed_simulator": seed}

    switcher = {
        "ideal": lambda: config_ideal,
    }
    args["config"] = switcher.get("ideal", lambda: None)()

    _, test_loader = data_processing.get_data(args)
    test_features, test_labels = test_loader[0], test_loader[1]
    sig_fold, bkg_fold = data_processing.get_kfold_data(
        test_features, test_labels, 5
    )

    print("Computing model scores... ", end="")
    scores_time_init = perf_counter()

    if mode == CUSTOM:
        out_path = "trained_qsvms/model_testing"
    if mode == FIXED:
        out_path = "trained_qsvms/fixed_testing"
    if mode == CLASSIC:
        out_path = "trained_qsvms/classic_testing"

    out_path += "_L"+str(L)+"_T"+str(n_train)+"_P"+str(problem_id)+"/"

    print(f"Multiple k={5} folds...")
    score_sig = np.array([model.decision_function(fold) for fold in sig_fold])
    score_bkg = np.array([model.decision_function(fold) for fold in bkg_fold])
    scores_all = model.decision_function(test_features)
    print(
        f"Saving the signal and background k-fold scores in the folder: "
        + tcols.OKCYAN
        + f"{out_path}"
        + tcols.ENDC
    )
    np.save(
        out_path + f"sig_scores_n{args['ntest']}_k{args['kfolds']}.npy",
        score_sig,
    )
    np.save(
        out_path + f"bkg_scores_n{args['ntest']}_k{args['kfolds']}.npy",
        score_bkg,
    )

    if mode != CLASSIC:
        np.save(out_path + f"kernel_matrix_test.npy", model._kernel_matrix_test)
    scores_time_fina = perf_counter()
    exec_time = scores_time_fina - scores_time_init
    print(
        tcols.OKGREEN
        + "Completed in: "
        + tcols.ENDC
        + f"{exec_time:2.2e} sec. or {exec_time/60:2.2e} min. "
        + tcols.ROCKET
    )

    return None

  

def time_and_train(fit: Callable, *args):
    """Trains and computes the training runtime of the qsvm model.

    Parameters
    ----------
    fit : `Callable`
        Fitting function of the corresponding model.
    args: dict
        Arguments of the fit function.
    """
    print("Training the QSVM... ", end="")
    train_time_init = perf_counter()
    fit(*args)
    train_time_fina = perf_counter()

    exec_time = train_time_fina - train_time_init
    print(
        "Training completed in: " + tcols.OKGREEN + f"{exec_time:.2e} sec. "
        f"or {exec_time/60:.2e} min. " + tcols.ENDC + tcols.SPARKS
    )

