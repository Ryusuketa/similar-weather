import numpy as np


class Sampler(object):
    def __init__(self, sim_positive_max, sim_positive_min, sim_negative_max, sim_negative_min, n_samples=50000):
        self.sim_negative_max = sim_negative_max
        self.sim_negative_min = sim_negative_min
        self.sim_positive_max = sim_positive_max
        self.sim_positive_min = sim_positive_min
        self.n_samples = n_samples

    def sample(self, X, y):
        positive_index = np.where(np.logical_and(y > self.sim_positive_min, y < self.sim_positive_max))[0]
        negative_index = np.where(np.logical_and(y > self.sim_negative_min, y < self.sim_negative_max))[0]

        positive_sampling = np.random.choice(positive_index, self.n_samples)
        negative_sampling = np.random.choice(negative_index, self.n_samples)

        sampling = np.concatenate([positive_sampling, negative_sampling])

        X = X[sampling, :]
        y = np.concatenate([np.ones(self.n_samples), np.zeros(self.n_samples)])

        return X, y

    def correlation_to_binary(self, y):
        positive_index = np.where(np.logical_and(y > self.sim_positive_min, y < self.sim_positive_max))[0]
        negative_index = np.where(np.logical_and(y > self.sim_negative_min, y < self.sim_negative_max))[0]

        y_bin = np.zeros(len(y))
        y_bin[positive_index] = 1
        y_bin[negative_index] = 0

        return y_bin
