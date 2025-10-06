import gymnasium as gym
import numpy
import numpy as np

class AWActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Mapping from desired actions to raw MultiBinary(12)
        self._actions = []

        def press(idx):
            a = np.zeros(12, dtype=np.int8)
            a[idx] = 1
            return a

        # Single-button presses
        self._actions.append(press(8))   # A
        self._actions.append(press(0))   # B
        self._actions.append(press(4))   # UP
        self._actions.append(press(5))   # DOWN
        self._actions.append(press(6))   # LEFT
        self._actions.append(press(7))   # RIGHT
        self._actions.append(press(10))  # L


        self._macros = {
            8: [press(8), press(4), press(8)]  # END_TURN
        }

        self.action_names = ['A','B','UP','DOWN','LEFT','RIGHT','L','END_TURN']
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def step(self, action_idx):
        if isinstance(action_idx, numpy.ndarray):
            self.env.step(action_idx)
        elif action_idx in self._macros:
            macro = self._macros[action_idx]
            obs, reward, done, info = None, 0, False, {}
            for a in macro:
                obs, r, d, _, info = self.env.step(a)
            return obs, reward, done, _, info
        else:
            return self.env.step(self._actions[action_idx])
