import numpy as np
from ..core import Kernel
from . import KernelEvaluator

from qiskit import QuantumCircuit
from qiskit import Aer, transpile

from darqk.core import Ansatz, Kernel, KernelFactory, KernelType

from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
import joblib
from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from qiskit.providers.ibmq import IBMQBackend
from qiskit.visualization import plot_circuit_layout
from qiskit_machine_learning.kernels import QuantumKernel, FidelityQuantumKernel
import numpy as np


from .latent_ad_qml.scripts.kernel_machines.train_callable import train_and_evaluate


from .latent_ad_qml.scripts.kernel_machines.my_util import to_optimize_kernel

class OneClassSVMEvaluator(KernelEvaluator):
    """
    Kernel compatibility measure based on the kernel-target alignment
    See: Cristianini, Nello, et al. "On kernel-target alignment." Advances in neural information processing systems 14 (2001).
    
    La funzione da minimizzare è lo score dell'oggetto 'OneClassQSVM' del paper di Belis et al. 2023
    
    """

    def __init__(self, ntest, ntrain, L, problem_id):

        self.ntest = ntest
        self.ntrain = ntrain
        self.latent_dim = L
        self.problem_id = problem_id



    def evaluate(self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray, save = False):
        """
        Evaluate the current kernel and return the corresponding cost. Lower cost values corresponds to better solutions
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better

        mode = "FAST", "SLOW"

        """


        #print(kernel)

        qiskit_circuit = kernel.to_qiskit_circuit()

        simulator = Aer.get_backend('aer_simulator')    
        qiskit_circuit = transpile(qiskit_circuit, simulator)

        #print(qiskit_circuit)

        qiskit_kernel = FidelityQuantumKernel(feature_map=qiskit_circuit)




        #the_cost = train_and_evaluate(kernel = qiskit_kernel, mode = self.mode, nqubits = self.nqubits, save=save, path = self.path)
        the_cost = to_optimize_kernel(qiskit_kernel, self.ntrain, self.ntest, self.latent_dim, self.problem_id)

        print(the_cost)

        # assert not np.isnan(the_cost), f"{kernel=} {K=} {y=}"
        return -the_cost if not np.isnan(the_cost) else 1000


