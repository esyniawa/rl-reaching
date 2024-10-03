import numpy as np


def linear_reward(error, max_distance):
    return 1 - error / max_distance


def gaussian_reward(error: float, sigma: float, amplitude: float = 1.):
    # error and sigma should be in mm
    return amplitude * np.exp(-error**2 / (2 * sigma**2))


def logarithmic_reward(error, max_distance):
    return 1 - np.log(1 + error) / np.log(1 + max_distance)


def sigmoid_reward(error, max_distance, k=10):
    return 1 / (1 + np.exp(k * (error / max_distance - 0.5)))
