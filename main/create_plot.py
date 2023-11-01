import h5py
import argparse
from plot import plot_ROC_kfold_mean

def create_plot(file_name, mode, latent_dim):

    read_dir = [f"{file_name}_P2.h5", f"{file_name}_P0.h5", f"{file_name}_P1.h5"]

    n_folds = 5
    n_samples_train=600
    mass=['35', '35', '15']
    br_na=['NA', '', 'BR'] # narrow (NA) or broad (BR)
    signal_name=['RSGraviton_WW', 'AtoHZ_to_ZZZ', 'RSGraviton_WW']
    ntest = ['100', '100', '100']

    colors = ['forestgreen', '#EC4E20', 'darkorchid']

    L = latent_dim
    comp_mode = mode
    add = ""

    legend_signal_names=['Narrow 'r'G $\to$ WW 3.5 TeV', r'A $\to$ HZ $\to$ ZZZ 3.5 TeV', 'Broad 'r'G $\to$ WW 1.5 TeV']

    q_loss_qcd=[]; q_loss_sig=[]; c_loss_qcd=[]; c_loss_sig=[]
    for i in range(len(signal_name)):
        with h5py.File(read_dir[i], 'r') as file:
            q_loss_qcd.append(file['quantum_loss_qcd'][:])
            q_loss_sig.append(file['quantum_loss_sig'][:])
            c_loss_qcd.append(file['classic_loss_qcd'][:])
            c_loss_sig.append(file['classic_loss_sig'][:])

    name = "Discovered vs "
    other_name = "Dx"

    if comp_mode == "CLASSICAL":
        name += "Classical"
        other_name += "C"
    if comp_mode == "FIXED":
        name += "Fixed"
        other_name += "F"
    name += ", L="+str(L)
    other_name += "_L"+str(L)+add

    plot_ROC_kfold_mean(q_loss_qcd, q_loss_sig, c_loss_qcd, c_loss_sig, legend_signal_names, n_folds,\
                    legend_title=r'Anomaly signature', save_dir='plots', pic_id=other_name, name = name,
                    comparison_mode = comp_mode,
                    palette=colors, xlabel=r'$TPR$', ylabel=r'$FPR^{-1}$')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file_name", type=str, required=True, help="Base name for the files to be read.")
    parser.add_argument("--mode", type=str, required=True, choices=['CLASSICAL', 'FIXED'], help="Mode of operation. CLASSICAL or FIXED.")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension (L).")
    args = parser.parse_args()

    create_plot(args.file_name, args.mode, args.latent_dim)
