# The quantum and classical kernel machine training script.
# The model is instantiated with some parameters, the data encoding circuit is built,
# it is trained on a data set, and is saved in a folder.

import argparse
import json
from time import perf_counter
from typing import Callable
from qiskit.utils import algorithm_globals
import os

import numpy as np

import qad.algorithms.kernel_machines.util as util
import qad.algorithms.kernel_machines.data_processing as data_processing
from qad.algorithms.kernel_machines.terminal_enhancer import tcols

from qiskit_machine_learning.kernels import FidelityQuantumKernel

seed = 12345
algorithm_globals.random_seed = seed

#params_def:

basic_path = "QML_paper_data/latent4/"

sig_path = "QML_paper_data/latent4/latentrep_AtoHZ_to_ZZZ_35.h5"
bkg_path = "QML_paper_data\latent4\latentrep_QCD_sig.h5"
test_bkg_path = "QML_paper_data/latent4/latentrep_QCD_sig_testclustering.h5"

p1 = basic_path+"latentrep_AtoHZ_to_ZZZ_35.h5"
p2 = basic_path+"latentrep_RSGraviton_WW_BR_15.h5"
p3 = basic_path+"latentrep_RSGraviton_WW_NA_35.h5"

nqubits = 4

unsup = True
feature_map = "u_dense_encoding"
run_type = "ideal"
output_folder = "quantum_test"
nu_param = 0.01
ntrain = 60 #60 #ridefiniti dopo
ntest = 720
quantum = True


def train_and_evaluate(kernel = None, mode = "FAST", nqubits = 8, save = False, path = None):
    config_ideal = {"seed_simulator": seed}

    switcher = {
        "ideal": lambda: config_ideal,
    }

    if mode == "FAST":
        ntrain = 10
        ntest = 120
    else:
        ntrain = 60
        ntest = 720       

    args = {
        "sig_path": p3,
        "bkg_path": bkg_path,
        "test_bkg_path": test_bkg_path,
        "c_param": 1.0,
        "nu_param": nu_param,
        "output_folder": output_folder,
        "gamma": "scale",
        "quantum": quantum,
        "unsup": True,
        "nqubits": nqubits,
        "feature_map": feature_map,
        "backend_name": "statevector_simulator", #args.backend_name,
        "ibmq_api_config": None, #private_configuration["IBMQ"],
        "run_type": "ideal",
        #"config": config,
        "ntrain": ntrain,
        "ntest": ntest, #720!!!
        "seed": seed,  # For the data shuffling.
    }

    args["config"] = switcher.get("ideal", lambda: None)()



    train_loader, test_loader = data_processing.get_data(args, print_info=False)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]
    
    if kernel == None: ###!!!!!!!
        model = util.init_kernel_machine(args)
    else:
        #print("I am loading my kernel\n")
        model = util.init_kernel_machine(args, kernel=kernel, use_custom_kernel=True)

    #time_and_train(model.fit, train_features, train_labels)

    ntrain = 10
    ntest = 120


    F = "_L4"
    O = "_Op18"
    t = "_tr"+str(ntrain)
    T = "_te"+str(ntest)
    d = "_P3"
    q = "_Q"+str(args["nqubits"])

    out_path = "trained_qsvms/model"+F+O+t+T+d+q

    ntrain = 60
    ntest = 720

    time_and_train(model.fit, train_features, train_labels)

    print("Just for testing, no training + ")

    if True: #save

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        #print("Doing the last training:\nntrain set to 600.")
        #args["ntrain"] = 600
        #time_and_train(model.fit, train_features, train_labels)
        #util.print_model_info(model)
        #util.export_hyperparameters(model, out_path) #questo non fa niente
        #util.save_model(model, out_path)

        
        test_callable(mode=mode, nqubits=nqubits, path=out_path, ntest=1500)

    else:
        print("Sistema...")

    if args["run_type"] != "hardware":
        test_acc_value = util.eval_metrics(
            model, train_features, train_labels, test_features, test_labels, out_path
    )
    else:
        test_acc_value = 0
    
    return test_acc_value
    

def time_and_train(fit: Callable, *args):
    """Trains and computes the training runtime of the qsvm model.

    Parameters
    ----------
    fit : `Callable`
        Fitting function of the corresponding model.
    args: dict
        Arguments of the fit function.
    """
    #print("Training the QSVM... ", end="")
    train_time_init = perf_counter()
    fit(*args)
    train_time_fina = perf_counter()

    exec_time = train_time_fina - train_time_init
    print(
        "Training completed in: " + tcols.OKGREEN + f"{exec_time:.2e} sec. "
        f"or {exec_time/60:.2e} min. " + tcols.ENDC + tcols.SPARKS
    )

#def train_from_kernel()

def get_arguments() -> dict:
    args = None
    raise Exception("QUESTO NON DEVE ESSERE CHIAMATO...")
    return args

#my_val = train_and_evaluate(kernel=None)
#print("Dai che ci siamo:\n"+str(my_val))

def test_callable(mode = "FAST", nqubits = 0, path = None, ntest = 1500):
    seed = 12345
    config_ideal = {"seed_simulator": seed}

    switcher = {
        "ideal": lambda: config_ideal,
    }


    ntrain = 600 #non c'è training!!
          #<- forse anche di più

    args = {
        "sig_path": p1,
        "bkg_path": bkg_path,
        "test_bkg_path": test_bkg_path,
        "c_param": 1.0,
        "nu_param": nu_param,
        "output_folder": output_folder,
        "gamma": "scale",
        "quantum": quantum,
        "unsup": True,
        "nqubits": nqubits,
        "feature_map": feature_map,
        "backend_name": "statevector_simulator", #args.backend_name,
        "ibmq_api_config": None, #private_configuration["IBMQ"],
        "run_type": "ideal",
        #"config": config,
        "ntrain": ntrain,
        "ntest": ntest, #720!!!
        "seed": seed,  # For the data shuffling.
        "kfolds": 5,
        "model": "trained_qsvms/quantum_test_nu=0.01_ideal_statevector_simulator/"
    }

    args["config"] = switcher.get("ideal", lambda: None)()
    _, test_loader = data_processing.get_data(args)
    test_features, test_labels = test_loader[0], test_loader[1]
    sig_fold, bkg_fold = data_processing.get_kfold_data(
        test_features, test_labels, args["kfolds"]
    )
    output_path = path+"/"
    model = util.load_model(output_path+"model")

    seed = 12345

    print("Computing model scores... ", end="")
    scores_time_init = perf_counter()

    if args["kfolds"] == 1:
        print("Only one fold...")
        if args["mod_quantum_instance"]:
            raise Exception("Non dovrebbe essere possibile chiamare questo pezzo di codice...")
        scores = model.decision_function(test_features)
        np.save(output_path + f"scores_n{args['ntest']}_k{args['kfolds']}.npy", scores)
        np.save(output_path + f"kernel_matrix_test.npy", model._kernel_matrix_test)
        scores_time_fina = perf_counter()
    else:
        print(f"Multiple k={args['kfolds']} folds...")
        score_sig = np.array([model.decision_function(fold) for fold in sig_fold])
        score_bkg = np.array([model.decision_function(fold) for fold in bkg_fold])
        scores_all = model.decision_function(test_features)
        print(
            f"Saving the signal and background k-fold scores in the folder: "
            + tcols.OKCYAN
            + f"{output_path}"
            + tcols.ENDC
        )
        np.save(
            output_path + f"sig_scores_n{args['ntest']}_k{args['kfolds']}.npy",
            score_sig,
        )
        np.save(
            output_path + f"bkg_scores_n{args['ntest']}_k{args['kfolds']}.npy",
            score_bkg,
        )

        if True: #isinstance(model, OneClassQSVM):
            np.save(output_path + f"kernel_matrix_test.npy", model._kernel_matrix_test)
        scores_time_fina = perf_counter()
    exec_time = scores_time_fina - scores_time_init
    print(
        tcols.OKGREEN
        + "Completed in: "
        + tcols.ENDC
        + f"{exec_time:2.2e} sec. or {exec_time/60:2.2e} min. "
        + tcols.ROCKET
    )



