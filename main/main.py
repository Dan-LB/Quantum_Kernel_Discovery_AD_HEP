import argparse
import os
import random
import numpy as np
from qiskit import Aer, transpile
from darqk.core import Ansatz, KernelFactory, KernelType
from darqk.optimizer import BayesianOptimizer
from darqk.evaluator.one_classSVM_evaluator import OneClassSVMEvaluator
from darqk.evaluator.latent_ad_qml.scripts.kernel_machines.my_util import (save_kernel,
                                                                           create_model_from_qiskit_circuit,
                                                                           create_model_with_defaul_kernel,
                                                                           create_model_classic,
                                                                           train_model,
                                                                           test_model)

# Define the command-line argument parser
def define_parser():
    parser = argparse.ArgumentParser(description='Train and Test Quantum Models')
    parser.add_argument('mode', type=str, choices=['CUSTOM', 'FIXED', 'CLASSICAL'],
                        help='Mode of training: CUSTOM, FIXED, or CLASSICAL')
    parser.add_argument('--latent_dim', type=int, choices=[4, 8, 16, 32],
                        help='Latent dimension (required for CUSTOM mode)')
    parser.add_argument('--opt_param', type=int, help='Optimization parameter (required for CUSTOM mode)')
    parser.add_argument('--n_op', type=int, help='Number of operations (required for CUSTOM mode)')
    parser.add_argument('--ntest', type=int, default=75, help='Number of tests (default: 75)')
    parser.add_argument('--ntrain', type=int, default=75, help='Number of trainings (default: 75)')
    parser.add_argument('--final_train', type=int, default=200, help='Final training size (default: 200)')
    parser.add_argument('--final_test', type=int, default=1500, help='Final test size (default: 1500)')
    return parser

# Set seeds for reproducibility
def set_seeds():
    random.seed(12345)
    np.random.seed(12345)

# Main function that integrates your logic
def main(args):
    set_seeds()

    for p in [0, 1, 2]:

    # Mode-specific logic
        if args.mode == 'CUSTOM':
            # Your CUSTOM mode logic
            ansatz = Ansatz(n_features=args.latent_dim*2, n_qubits=args.latent_dim, n_operations=args.n_op)
            ansatz.initialize_to_identity()
            kernel = KernelFactory.create_kernel(ansatz, "Z" * args.latent_dim, KernelType.OBSERVABLE)
            ke = OneClassSVMEvaluator(args.ntest, args.ntrain, args.latent_dim, p) # `p` parameter not defined in args, defaulting to 1
            bayesian_opt = BayesianOptimizer(kernel, None, None, ke)  # X and Y not defined in args
            optimized_kernel = bayesian_opt.optimize(n_epochs=args.opt_param, n_points=5, n_jobs=1)
            save_as = "kernel_L"+str(args.latent_dim)+"_P"+str(p)  # `p` parameter hardcoded to 1
            save_kernel(optimized_kernel, save_as)
            qiskit_c = optimized_kernel.to_qiskit_circuit()
            simulator = Aer.get_backend('aer_simulator')
            transpiled_c = transpile(qiskit_c, simulator)
            model, args_ = create_model_from_qiskit_circuit(transpiled_c, args.latent_dim)
            train_model(model, args_, args.final_train, 100, args.latent_dim, p, mode=args.mode)  # `p` hardcoded to 1
            test_model(model, args.final_test, args.latent_dim, p, n_train=args.final_train, mode=args.mode)  # `p` hardcoded to 1

        elif args.mode == 'FIXED':
            # Your FIXED mode logic
            model, args_ = create_model_with_defaul_kernel(args.latent_dim)
            train_model(model, args_, args.final_train, 100, args.latent_dim, p, mode=args.mode)  # `p` hardcoded to 1
            test_model(model, args.final_test, args.latent_dim, p, n_train=args.final_train, mode=args.mode)  # `p` hardcoded to 1

        elif args.mode == 'CLASSICAL':
            # Your CLASSIC mode logic
            model, args_ = create_model_classic(args.latent_dim)
            train_model(model, args_, args.final_train, 100, args.latent_dim, p, mode=args.mode)  # `p` hardcoded to 1
            test_model(model, args.final_test, args.latent_dim, p, n_train=args.final_train, mode=args.mode)  # `p` hardcoded to 1

# Run the script with command-line arguments
if __name__ == '__main__':
    parser = define_parser()
    args = parser.parse_args()
    main(args)
