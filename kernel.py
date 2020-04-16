from math import exp
import numpy as np
import configuration


def linear_kernel(descr_list1: np.array, descr_list2: np.array) -> float:
    if np.shape(descr_list1)[0] != np.shape(descr_list2.T)[0]:
        raise ValueError('Shapes of input do not match')

    return descr_list1.dot(descr_list2.T)


def gaussian_kernel(descriptor1: np.array, descriptor2: np.array, sigma: float) -> float:
    if np.shape(descriptor1) != np.shape(descriptor2):
        raise ValueError('Shapes of input do not match')

    dr = descriptor1 - descriptor2
    return exp(dr.dot(dr) / (2 * sigma**2))


# returns the scalar prefactor for the matrix element of the forces
def grad_scalar(q: float, dr: np.array) -> np.array:
    return q * np.cos(q * dr) / dr


# builds part of the row for the force kernel matrix given a configuration and a set of descriptors
def linear_force_submat(q: np.array, config1: configuration, descriptors_array: np.array) -> np.array:
    nr_modi = len(q)
    n_ions, modi_config = np.shape(config1.descriptors)
    _, dim = np.shape(config1.positions)
    nr_descriptors, modi_desc = np.shape(descriptors_array)

    if not nr_modi == modi_config == modi_desc:
        raise ValueError('The nr of q\'s does not match')
    
    submat = np.zeros((n_ions * dim, nr_descriptors))

    for l in range(nr_modi):
        # build the scalar prefactor for each distance vector
        summands = np.zeros((n_ions, dim))
        for k in range(n_ions):
            factor = grad_scalar(q[l], np.array(config1.nndistances[k]))
            summands[k] = np.sum(
                factor[:, np.newaxis] * np.array(config1.nndisplacements[k]),
                axis=0
            )

        submat += (descriptors_array[:, l, np.newaxis] * summands.flatten()).T

    return -2 * submat


class Kernel:
    def __init__(self, mode, *args):
        if mode == 'linear':
            self.kernel = linear_kernel
            self.force_mat = linear_force_submat
        elif mode == 'gaussian':
            if not args:
                raise ValueError('For the Gaussian Kernel a sigma has to be supplied')
            self.kernel = lambda x, y: gaussian_kernel(x, y, args[0])
        else:
            raise ValueError(f'kernel {mode} is not supported')

    # builds a matrix-element for a given configuration
    # and !!one!! given descriptor vector (i.e. for !!one!! atom)
    def energy_matrix_element(self, config: configuration, descriptors_array: np.array) -> float:
        return sum(self.kernel(descriptors_array, config.descriptors))

    # builds part of the row of the energy kernel matrix
    def energy_subrow(self, config1: configuration, descriptors_array: np.array) -> np.array:
        return np.apply_along_axis(
            lambda x: self.energy_matrix_element(config1, x),
            arr=descriptors_array,
            axis=1
        )


    # applies the correct function to build the force submatrix
    def force_submat(self, q: np.array, config1: configuration, config2: configuration) -> np.array:
        return self.force_mat(q, config1, config2.descriptors)