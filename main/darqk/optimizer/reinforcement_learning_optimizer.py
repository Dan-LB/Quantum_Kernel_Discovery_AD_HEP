import numpy as np
import copy
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete
from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import SARSALambda
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J
from ..core import Operation, Ansatz, Kernel, KernelFactory
from ..evaluator import KernelEvaluator
from . import BaseKernelOptimizer

from .tight_kernel_environment import TightKernelEnvironment
from .wide_kernel_environment import WideKernelEnvironment

class ReinforcementLearningOptimizer(BaseKernelOptimizer):
    """
    Reinforcement learning based technique for optimize a kernel function
    """

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator, env = "TIGHT", bw_possible = 10):
        """
        Initialization
        :param initial_kernel: initial kernel object
        :param X: datapoints
        :param y: labels
        :param ke: kernel evaluator object
        """
        self.initial_kernel = copy.deepcopy(initial_kernel)
        self.X = X
        self.y = y
        self.ke = ke

        self.bw = bw_possible
        if env == "TIGHT":
            #from tight_kernel_environment import TightKernelEnvironment
            self.mdp: TightKernelEnvironment = Environment.make('TightKernelEnvironment', initial_kernel=self.initial_kernel, X=X, y=y, ke=ke, bw_possible = bw_possible, convert_to_int = True)
        if env == "WIDE":
            #from . import WideKernelEnvironment
            self.mdp: WideKernelEnvironment = Environment.make('WideKernelEnvironment', initial_kernel=self.initial_kernel, X=X, y=y, ke=ke, bw_possible = bw_possible, convert_to_int = True)
        self.agent = None
        self.core = None

    def optimize(self, initial_episodes=3, n_episodes=100, n_steps_per_fit=1, final_episodes=3):
        """
        Optimization routine
        :param initial_episodes:
        :param n_steps:
        :param n_steps_per_fit:
        :param final_episodes:
        :return:
        """

        # Policy
        #epsilon???
        epsilon = Parameter(value=1) #forse meglio 0.1, valore iniziale: 1
        # qual è il decay value, se presente?
        pi = EpsGreedy(epsilon=epsilon)
        learning_rate = Parameter(.1)

        # Agent
        self.agent = SARSALambda(self.mdp.info, pi,
                            learning_rate=learning_rate,
                            lambda_coeff=.9)

        # Reinforcement learning experiment
        self.core = Core(self.agent, self.mdp)

        print(self.mdp)
        print(self.mdp.info)
        print(self.core)

        # Visualize initial policy for 3 episodes
        dataset = self.core.evaluate(n_episodes=initial_episodes, render=True)
        print(f"{dataset=}")

        # Print the average objective value before learning
        J = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f'Objective function before learning: {J}')

        # Train
        #qua smette di funzionare!!!!!!!

        self.core.learn(n_episodes=n_episodes, n_steps_per_fit=n_steps_per_fit, render=True)

        # Visualize results for 3 episodes
        dataset = self.core.evaluate(n_episodes=final_episodes, render=True)

        # Print the average objective value after learning
        J = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f'Objective function after learning: {J}')

        #celle corrispondenti al bw [4?] diviso per b


        print("Print per assicurarmi che il codice per la bandwidth funzioni bene:")
        print("Pre divisione:")
        array = self.mdp._state.astype(float)
        print(array)

        for i in range(self.mdp.n_operations):
            array[5+i*5] /= self.bw

        print("Post divisione:")
        print(array)
        kernel = Kernel.from_numpy(array[1:], self.mdp.n_features, self.mdp.n_qubits, self.mdp.n_operations, self.mdp.allow_midcircuit_measurement)
        #kernel = Kernel.from_numpy(self.mdp._state[1:], self.mdp.n_features, self.mdp.n_qubits, self.mdp.n_operations, self.mdp.allow_midcircuit_measurement)
        return kernel
