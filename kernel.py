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


def derivatives(q: np.array, displ: np.array, dist: np.array) -> np:
    '''
    Calculate the derivatives matrix of the descriptors
    '''
    nq = len(q)
    nc, ni, _, d = displ.shape
    rq = dist.reshape(nc, ni, ni, 1) * q
    qcosrq = q * np.cos(rq)
    R_over_r = displ / dist.reshape(nc, ni, ni, 1)
    # filter out where we divide by 0 and replace by 0
    R_over_r[np.isnan(R_over_r)] = 0
    D = qcosrq.reshape(nc, ni, ni, 1, nq) *  R_over_r.reshape(nc, ni, ni, d, 1)

    return D


# builds part of the row for the force kernel matrix given a configuration and a set of descriptors
def linear_force_submat(q: np.array, config1: configuration, descriptors_array: np.array) -> np.array:
    nr_modi = len(q)
    n_ions, modi_config = np.shape(config1.descriptors)
    _, dim = np.shape(config1.positions)
    nr_descriptors, modi_desc = np.shape(descriptors_array)

    if not nr_modi == modi_config == modi_desc:
        raise ValueError('The nr of q\'s does not match')

    submat = np.zeros((n_ions * dim, nr_descriptors))

    # IDEA: multiproccessing/ray to parallelize this loop -> interchange loops to have the
    # longer loop para.
    # WARNING: Only the outer loop
    for l in range(nr_modi):
        # build the scalar prefactor for each distance vector
        summands = np.zeros((n_ions, dim))
        for k in range(n_ions):
            factor = grad_scalar(q[l], config1.nndistances[k])
            summands[k] = np.sum(factor[:, np.newaxis] * config1.nndisplacements[k],axis=0)

        submat += (descriptors_array[:, l, np.newaxis] * summands.flatten()).T

    return -2 * submat

def gaussian_force_submat(q: np.array, config1: configuration, descriptors_array: np.array, sig: float) -> np.array:
    nr_modi = len(q)
    n_ions, modi_config = np.shape(config1.descriptors)
    _, dim = np.shape(config1.positions)
    nr_descriptors, modi_desc = np.shape(descriptors_array)

    if not nr_modi == modi_config == modi_desc:
        raise ValueError('The nr of q\'s does not match')

    submat = np.zeros((n_ions, dim, nr_descriptors))

    for l in range(nr_modi):
        for k in range(n_ions):
            factor1 = grad_scalar(q[l], config1.nndistances[k])[:, np.newaxis] * config1.nndisplacements[k]

            factor3a = gaussian_kernel(config1.get_nndescriptor(k), descriptors_array, sig)
            factor4a = descriptors_array[np.newaxis,:,l] - config1.get_nndescriptor(k)[:,l,np.newaxis]
            factor3b = gaussian_kernel(config1.descriptors[k,:], descriptors_array, sig)
            factor4b = descriptors_array[np.newaxis,:,l] - config1.descriptors[k,l,np.newaxis]
            factor5 = factor3a*factor4a + factor3b*factor4b

            factor_all = np.sum(factor1[:, np.newaxis, :] * factor5[:, :, np.newaxis], axis=0)
            submat[k,:,:] += factor_all.T

    submat = np.reshape(submat, (n_ions * dim, nr_descriptors))
    return -1/sig**2 * submat

class Kernel:
    def __init__(self, mode, *args):
        if mode == 'linear':
            self.kernel = linear_kernel
            self.force_mat = linear_force_submat
        elif mode == 'gaussian':
            if not args:
                raise ValueError('For the Gaussian Kernel a sigma has to be supplied')
            self.kernel = lambda x, y: gaussian_kernel(x, y, args[0])
            self.force_mat = lambda x, y, z: gaussian_force_submat(x, y, z, args[0])
        else:
            raise ValueError(f'kernel {mode} is not supported')


    # builds a matrix-element for a given configuration
    # and !!one!! given descriptor vector (i.e. for !!one!! atom)
    #def energy_matrix_elements(self, descr1: np.array, descr2: np.array) -> np.array:
    #    # TODO: check sum
    #    # FIXME: this does not play well, if we want to completely flatten both input arrays.
    #    sums = np.sum(self.kernel(descr1, descr2), axis=0)
    #    return sums

    # applies the correct function to build the force submatrix
    #def force_submat(self, q: np.array, config1: configuration, config2: configuration) -> np.array:
    #    return self.force_mat(q, config1, config2.descriptors)
