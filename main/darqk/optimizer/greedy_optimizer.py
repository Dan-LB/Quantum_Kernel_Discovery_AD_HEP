import copy

import numpy as np
from mushroom_rl.core import Environment

from ..core import Kernel, KernelFactory, KernelType
from ..evaluator import KernelEvaluator
from . import BaseKernelOptimizer


from .tight_kernel_environment import TightKernelEnvironment
from .wide_kernel_environment import WideKernelEnvironment

class GreedyOptimizer(BaseKernelOptimizer):

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator, env = "TIGHT", bw_possible = 1):
        super().__init__(initial_kernel, X, y, ke)
        if env == "TIGHT":
            #from tight_kernel_environment import TightKernelEnvironment
            self.mdp: TightKernelEnvironment = Environment.make('TightKernelEnvironment', initial_kernel=self.initial_kernel, X=X, y=y, ke=ke, bw_possible = bw_possible)
        if env == "WIDE":
            #from . import WideKernelEnvironment
            self.mdp: WideKernelEnvironment = Environment.make('WideKernelEnvironment', initial_kernel=self.initial_kernel, X=X, y=y, ke=ke, bw_possible = bw_possible)
        self.rewards_history = []
        self.actions_history = []




    def greedy_measure(self, kernel):
        meas = ['X', 'Y', 'Z']
        string = kernel.measurement
        print("Initial measurement: ", string)
        ke = self.mdp.ke
        X = self.mdp.X
        y = self.mdp.y

        initial_kernel = self.mdp.initial_kernel
        n_qubits = initial_kernel.ansatz.n_qubits

        best_value = ke.evaluate(kernel, None, X, y)
        print(best_value)

        for i in range(n_qubits):
            for m in meas:

                lst = list(string)
                lst[i] = m
                new_string = "".join(lst)

                #kernel.measurment = new_string

                kernel = KernelFactory.create_kernel(kernel.ansatz, new_string, KernelType.OBSERVABLE)
                new_value = ke.evaluate(kernel, None, X, y)
                if new_value < best_value:
                    best_value = new_value
                    print("Nuova misura:", new_string)
                    string = new_string
                else:
                    #kernel.measurment = string
                    kernel = KernelFactory.create_kernel(kernel.ansatz, string, KernelType.OBSERVABLE) 
        return kernel   

    def optimize(self, greedy_measure=True, verbose=False):

        #self.mdp.reset()
        print("Attenzione: non stiamo resettando")
        state = copy.deepcopy(self.mdp._state)

        terminated = False
        n_actions = self.mdp._mdp_info.action_space.size[0]
        rewards = np.zeros(shape=(n_actions,))

        counter = 1


        while not terminated:
            print("COUNTER: ", str(counter))
            # list all actions at the first depth
            for action in range(n_actions):
                self.mdp.reset(state)
                new_state, reward, absorbed, _ = self.mdp.step((action,))
                rewards[action] = reward
                _, kernel = self.mdp.deserialize_state(new_state)
                if absorbed:
                    terminated = True
                #if verbose:
                print(f"{action=:4d} {reward=:0.6f} {kernel=}")
            # apply chosen action
            chosen_action = np.argmax(rewards)

            # tra poco il greedy!!!

            self.mdp.reset(state)
            state, _, _, _ = self.mdp.step((chosen_action,))
            

            # devo estrarre il kernel dallo stato, modificarlo e reinserirlo

            if greedy_measure == True:
                _, kernel = self.mdp.deserialize_state(state)
                new_kernel = self.greedy_measure(kernel)
                state = self.mdp.serialize_state(counter, new_kernel)


            if verbose:
                print(f"Chosen action: {chosen_action}")
                print(f"{self.mdp.deserialize_state(state)=}")
            # additional information
            self.rewards_history.append(rewards)
            self.actions_history.append(chosen_action)

            counter += 1

        _, kernel = self.mdp.deserialize_state(state)
        return kernel
