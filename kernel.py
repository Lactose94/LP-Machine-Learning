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
    def energy_subrow(self, config1: configuration, config2: configuration) -> np.array:
        return np.apply_along_axis(
            lambda x: self.energy_matrix_element(config1, x),
            arr=config2.descriptors,
            axis=1
        )

    # TODO: implement different kernels
    # builds part of the row for the force kernel matrix
    def force_subrow(self, q: np.array, config1: configuration, config2: configuration) -> np.array:
        nr_modi = len(q)
        Nions, len_desc = np.shape(config1.descriptors)
        _, dim = np.shape(config1.positions)
        if nr_modi != len_desc:
            raise ValueError('Dimension of supplied q and implied q by configuration1 do not match')
        
        submat = np.zeros((Nions * dim, Nions))
        # iterate over the i index. i.e. the atoms in config2

        for l in range(nr_modi):
            # build the scalar prefactor for each distance vector
            factors = grad_scalar(q[l], config1.distances)
            np.fill_diagonal(factors, 0)

            # multiply the distance vectors by their corresponding prefactor
            summands = factors[:, :, np.newaxis] * config1.differences

            # summ over all ions
            matrix_elements = np.sum(summands, axis=1)  
            #sum over nearest neighbors
            for i in range(Nions):
                matrix_elements[i] += np.sum(summands[config1.NNlist[i]], axis=0)
            # TODO: check this again
            vec_and_mat = config2.descriptors[:, l, np.newaxis, np.newaxis] * matrix_elements

            submat -= vec_and_mat.reshape(Nions, Nions * dim).T


        return submat
