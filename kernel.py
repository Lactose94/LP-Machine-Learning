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


# returns the scalar prefactor for the matrix element of the forces
def grad_scalar(q: float, dr: np.array) -> np.array:
    return q * np.cos(q * dr) / dr


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

    # builds a matrix-element for a given configuration
    # and !!one!! given descriptor vector (i.e. for !!one!! atom)
    def energy_matrix_element(self, config: configuration, descriptor: np.array) -> float:
        return sum(
            np.apply_along_axis(
                lambda x: self.kernel(descriptor, x),
                arr=config.descriptors,
                axis=1)
        )

    # builds part of the row of the energy kernel matrix
    # NOTE: This will be easy to refactor. Just replace config2 by set of descriptors
    def energy_subrow(self, config1: configuration, descriptors_array: np.array) -> np.array:
        return np.apply_along_axis(
            lambda x: self.energy_matrix_element(config1, x),
            arr=descriptors_array,
            axis=1
        )

    # TODO: implement different kernels
    # builds part of the row for the force kernel matrix
    # NOTE: this is also not to hard to rework. Again, replace config2 by the set of descriptors
    # TODO: In that case check the dimension for submat, as it surely has to change
    def force_subrow(self, q: np.array, config1: configuration, config2: configuration) -> np.array:
        nr_modi = len(q)
        n_ions, len_desc = np.shape(config1.descriptors)
        _, dim = np.shape(config1.positions)
        if nr_modi != len_desc:
            raise ValueError('Dimension of supplied q and implied q by configuration1 do not match')

        submat = np.zeros((n_ions * dim, n_ions))

        for l in range(nr_modi):
            # build the scalar prefactor for each distance vector
            summands = np.zeros((n_ions, dim))
            for k in range(n_ions):
                factor = grad_scalar(q[l], np.array(config1.nndistances[k]))
                summands[k] = np.sum(
                    factor[:, np.newaxis] * np.array(config1.nndisplacements[k]),
                    axis=0
                )

            submat += (config2.descriptors[:, l, np.newaxis] * summands.flatten()).T

        return -2 * submat
