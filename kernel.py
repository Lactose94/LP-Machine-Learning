from math import exp
import numpy as np
import configuration


def linear_kernel(descriptor1: np.array, descriptor2: np.array) -> float:
    if np.shape(descriptor1) != np.shape(descriptor2):
        raise ValueError('Shapes of input do not match')

    return np.inner(descriptor1, descriptor2)

def gaussian_kernel(descriptor1: np.array, descriptor2: np.array, sigma: float) -> float:
    if np.shape(descriptor1) != np.shape(descriptor2):
        raise ValueError('Shapes of input do not match')
    
    dr = descriptor1 - descriptor2
    return exp(dr.dot(dr) / (2 * sigma**2))


class Kernel:
    def __init__(self, mode, *args):
        if mode == 'linear':
            self.kernel = linear_kernel
        elif mode == 'gaussian':
            if not args:
                raise ValueError('For the Gaussian Kernel a sigma has to be supplied')
            self.kernel = lambda x, y: gaussian_kernel(x, y, args[0])
        else: 
            raise ValueError(f'kernel {mode} is not supported')
    
    # builds a matrix-element for a given configuration and !!one!! given descriptor vector (i.e. for !!one!! atom)
    def matrix_element(self, config: configuration, descriptor: np.array) -> float:
        return sum(
            np.apply_along_axis(
                lambda x: self.kernel(config.descriptors, x),
                arr=descriptor,
                axis=1)
        )

    # builds part of the row of the kernel matrix
    def build_subrow(self, config1: configuration, config2: configuration) -> np.array:
        return np.apply_along_axis(
            lambda x: self.matrix_element(config1, x),
            arr=config2.descriptors,
            axis=1
        )