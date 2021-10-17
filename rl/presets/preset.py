
import numpy as np


# TODO: register custom presets
class Preset:
    # https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/gym_lib.py#L41-L46

    # CartPole
    CARTPOLE_MIN = np.array([-2.4, -5.0, -np.pi / 12.0, -np.pi * 2.0])
    CARTPOLE_MAX = np.array([2.4, 5.0, np.pi / 12.0, np.pi * 2.0])
    CARTPOLE_RANGE = (CARTPOLE_MIN, CARTPOLE_MAX)

    # MountainCar
    MOUNTAIN_CAR_MIN = np.array([-1.2, -0.07])
    MOUNTAIN_CAR_MAX = np.array([0.6, 0.07])
    MOUNTAIN_CAR_RANGE = (MOUNTAIN_CAR_MAX, MOUNTAIN_CAR_MAX)
