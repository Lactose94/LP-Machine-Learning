import numpy as np
import configuration


def linear_kernel(descr_list1: np.array, descr_list2: np.array) -> np.array:
    shape1 = np.shape(descr_list1)
    shape2 = np.shape(descr_list2.T)

    if bool(shape1) ^ bool(shape2):
        raise ValueError('cannot apply to float and array')
    elif (bool(shape1) ^ bool(shape2)) and not shape1[-1] == shape2[0]:
        raise ValueError(f'Shapes of input do not match: {np.shape(descr_list1)} vs {np.shape(descr_list2.T)}')

    return np.dot(descr_list1, descr_list2.T)


# TODO: write decomposition into the mathematical documentation
# TODO: check if something's up with the diagonal elements
def gaussian_kernel(descr_list1: np.array, descr_list2: np.array, sigma: float) -> np.float:

    abs1 = linear_kernel(descr_list1, descr_list1)
    if not np.size(abs1) == 1:
        abs1 = np.diag(abs1)[:, np.newaxis]

    abs2 = linear_kernel(descr_list2, descr_list2)
    if not np.size(abs2) == 1:
        abs2 = np.diag(abs2)[np.newaxis, :]
    coeffs = linear_kernel(descr_list1, descr_list2)

    dr = abs1 - 2 * coeffs + abs2
    return np.exp(dr / (2 * sigma**2))


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
        # IDEA: rewrite the next few lines, s.t. it calculates the forces directly?
        summands = np.zeros((n_ions, dim))
        for k in range(n_ions):
            # IDEA: do not cast to array
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
            # TODO: Put in correct value for the gaussian force!
            self.force_mat = lambda x: x
        else:
            raise ValueError(f'kernel {mode} is not supported')

    # builds a matrix-element for a given configuration
    # and !!one!! given descriptor vector (i.e. for !!one!! atom)
    def energy_matrix_elements(self, descr1: np.array, descr2: np.array) -> np.array:
        # TODO: check sum
        # FIXME: this does not play well, if we want to completely flatten both input arrays.
        sums = np.sum(self.kernel(descr1, descr2), axis=0)
        return sums

    # applies the correct function to build the force submatrix
    def force_submat(self, q: np.array, config1: configuration, config2: configuration) -> np.array:
        return self.force_mat(q, config1, config2.descriptors)
