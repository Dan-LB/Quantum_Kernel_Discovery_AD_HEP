import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete
from ..core import Operation, Ansatz, Kernel, KernelFactory
from ..evaluator import KernelEvaluator


class WideKernelEnvironment(Environment):
    """
    Implementation of a Mushroom-RL Environment for our problem
    """

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator, bw_possible = 1, convert_to_int = False):
        """
        Initialization
        :param initial_kernel: initial kernel object
        :param X: datapoints
        :param y: labels
        :param ke: kernel evaluator object
        """
        self.initial_kernel = initial_kernel
        self.n_operations = self.initial_kernel.ansatz.n_operations
        self.n_features = self.initial_kernel.ansatz.n_features
        self.n_qubits = self.initial_kernel.ansatz.n_qubits
        self.allow_midcircuit_measurement = self.initial_kernel.ansatz.allow_midcircuit_measurement
        self.X = X
        self.y = y
        self.ke = ke
        self.last_reward = None

        self.bw = bw_possible

        # Create the action space.

        #['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']

        

        action_space = Discrete(
            len(self.initial_kernel.get_allowed_operations())
            * self.n_qubits
            * (self.n_qubits - 1)
            * (self.n_features + 1)
            * self.bw
        )

        # Create the observation space.
        observation_space = Discrete(
            len(self.initial_kernel.get_allowed_operations())
            * self.n_qubits
            * (self.n_qubits - 1)
            * (self.n_features + 1)
            * self.n_operations
        )

        # Create the MDPInfo structure, needed by the environment interface
        mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=100)
        super().__init__(mdp_info)

        # Create a state class variable to store the current state
        self._state = self.serialize_state(0, initial_kernel)

        # Create the viewer
        self._viewer = None

    def serialize_state(self, n_operation, kernel):
        """
        Pack the state of the optimization technique
        :param n_operation: number of operations currently performed
        :param kernel: kernel object
        :return: serialized state
        """
        state = np.concatenate([np.array([n_operation], dtype=int), kernel.to_numpy()], dtype=object).ravel() #loat invece di int

        if self.convert_to_int:
            for i in range(self.n_operations):
                if state[5+i*5] < 1:
                    state[5+i*5] *= self.bw

            return state.astype(int) #
    
        else:
             return state

    def deserialize_state(self, array):
        """
        Deserialized a previously packed state variable
        :param array: serialized state
        :return: tuple n_operations, kernel object
        """

        array = array.astype(float)
        if self.convert_to_int:
            for i in range(self.n_operations):
                array[5+i*5] = float(array[5+i*5])
                array[5+i*5] /= self.bw

        kernel = Kernel.from_numpy(array[1:], self.n_features, self.n_qubits, self.n_operations, self.allow_midcircuit_measurement)
        n_operations = int(array[0])

        return n_operations, kernel

    def render(self):
        """
        Rendering function - we don't need that
        :return: None
        """
        n_op, kernel = self.deserialize_state(self._state)
        print(f"{self.last_reward=:2.4f} {n_op=:2d} {kernel=}")

    def reset(self, state=None):
        """
        Reset the state
        :param state: optional state
        :return: self._state variable
        """
        if state is None:
            self.initial_kernel.ansatz.initialize_to_identity()
            self._state = self.serialize_state(0, self.initial_kernel)
        else:
            self._state = state
        return self._state

    def unpack_action(self, action):
        """
        Unpack an action to a operation
        :param action: integer representing the action
        :return: dictionary of the operation
        """
        generator_index = int(action % len(self.initial_kernel.get_allowed_operations()))
        action = action // len(self.initial_kernel.get_allowed_operations())

        wires_0 = int(action % self.n_qubits)
        action = action // self.n_qubits

        wires_1 = int(action % (self.n_qubits - 1))
        # perché questo???
        if wires_1 >= wires_0:
            wires_1 += 1
        action = action // (self.n_qubits - 1)

        feature = int(action % (self.n_features + 1))
        action = action // (self.n_features + 1)

        bw = int(action % self.bw)
        if self.convert_to_int:
            bandwidth =  bw+1#/self.bw
        else:
            bandwidth = float(bw+1)/self.bw
        action = action // self.bw

        assert action == 0
        
        return {'generator': self.initial_kernel.get_allowed_operations()[generator_index],
                'wires': [wires_0, wires_1],
                'feature': feature,
                'bandwidth': bandwidth}

        #?????????????????
        # bandwidth fissato???

    def step(self, action):

        the_action = self.unpack_action(action[0])

        # Create kernel from state
        n_operations, kernel = self.deserialize_state(self._state)

        # Update kernel
        kernel.ansatz.change_operation(n_operations, the_action['feature'], the_action['wires'], the_action['generator'], the_action['bandwidth'])
        n_operations += 1

        # Update state
        self._state = self.serialize_state(n_operations, kernel)

        # Compute the reward as distance penalty from goal
        reward = -1 * self.ke.evaluate(kernel, None, self.X, self.y)
        self.last_reward = reward

        # Set the absorbing flag if goal is reached
        absorbing = self.n_operations == n_operations

        # Return all the information + empty dictionary (used to pass additional information)
        return self._state, reward, absorbing, {}


WideKernelEnvironment.register()
