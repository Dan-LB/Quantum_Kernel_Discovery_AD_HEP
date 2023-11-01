import numpy as np
import argparse
import h5py

from qad.algorithms.kernel_machines.terminal_enhancer import tcols

SUFFIXES = ['P0', 'P1', 'P2']

def save_scores_h5(quantum_path: str, other_path: str, mode: str, out_path: str, name_suffix: str):
    for suffix in SUFFIXES:

        quantum_path_ = quantum_path + "_" + suffix + "/"
        other_path_ = other_path+ "_" + suffix + "/"
        
        print(f"Processing for suffix: {suffix}")

        print("Loading scores of the quantum model: " + tcols.OKBLUE + f"{quantum_path_}" + tcols.ENDC)

        quantum_sig = np.load(f"{quantum_path_}sig_scores_{name_suffix}.npy")
        quantum_bkg = np.load(f"{quantum_path_}bkg_scores_{name_suffix}.npy")

        print("Loading scores of the classical model: " + tcols.OKBLUE + f"{other_path_}" + tcols.ENDC)
        classical_sig = np.load(f"{other_path_}sig_scores_{name_suffix}.npy")
        classical_bkg = np.load(f"{other_path_}bkg_scores_{name_suffix}.npy")

        h5f = h5py.File(f"{out_path}_comparison_{mode.lower()}_{suffix}.h5", "w")
        h5f.create_dataset("quantum_loss_qcd", data=quantum_bkg)
        h5f.create_dataset("quantum_loss_sig", data=quantum_sig)
        h5f.create_dataset("classic_loss_qcd", data=classical_bkg)
        h5f.create_dataset("classic_loss_sig", data=classical_sig)
        h5f.close()

        print("Created a .h5 file containing the quantum and classical scores in: " + tcols.OKGREEN + f"{out_path}_comparison_{mode.lower()}_{suffix}" + tcols.ENDC)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--quantum_folder", type=str, required=True, help="Folder of the trained quantum model.")
    parser.add_argument("--comparison_folder", type=str, required=True, help="Folder of the trained  model to compare.")
    parser.add_argument("--mode", type=str, required=True, choices=['CLASSICAL', 'FIXED'], help="Mode of operation. Only CLASSICAL and FIXED are supported.")
    parser.add_argument("--out_path", type=str, required=True, help="Base path to the output file. Actual files will have the mode and suffix appended.")
    parser.add_argument("--name_suffix", type=str, required=True, help="String to append at the end of the sig_scores_<name_suffix>.npy and bkg_score<name_suffix>.npy output files.")
    args = parser.parse_args()

    save_scores_h5(args.quantum_folder, args.comparison_folder, args.mode, args.out_path, args.name_suffix)
