import numpy as np


class GaussianNoise:
    def __init__(self, size, mu=0, sigma=0.1):
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        return np.random.normal(self.mu, self.sigma, self.size)