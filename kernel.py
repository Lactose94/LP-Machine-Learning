import numpy as np
import configuration

# descr_list1 ist die aktuelle Konfiguration, descr_list2 die Referenz-Konfiguration!
def linear_kernel(descr_list1: np.array, descr_list2: np.array) -> np.array:
    shape1 = np.shape(descr_list1)
    shape2 = np.shape(descr_list2.T)

    if bool(shape1) ^ bool(shape2):
        raise ValueError('cannot apply to float and array')
    elif (bool(shape1) ^ bool(shape2)) and not shape1[-1] == shape2[0]:
        raise ValueError(f'Shapes of input do not match: {np.shape(descr_list1)} vs {np.shape(descr_list2.T)}')

    return np.matmul(descr_list1, descr_list2.T) # it says in the documentation that matmul is preferred over dot

# descr_list1 ist die aktuelle Konfiguration, descr_list2 die Referenz-Konfiguration!
def gaussian_kernel(descr_list1: np.array, descr_list2: np.array, sigma: float) -> np.array:

    nbnj, nq = np.shape(descr_list1)
    nani, _ = np.shape(descr_list2)

    abs1 = np.sum(descr_list1 ** 2, axis=1)
    abs2 = np.sum(descr_list2 ** 2, axis=1)

    coeffs = linear_kernel(descr_list1, descr_list2)

    dr = abs1.reshape(nbnj, 1) - 2 * coeffs + abs2.reshape(1, nani)

    return np.exp(-dr / (2 * sigma**2))


# builds part of the row for the force kernel matrix given a configuration and a set of descriptors
def linear_force_submat(q: np.array, config1: configuration, descriptors_array: np.array) -> np.array:
    '''
    Given one configuration and a set of descriptors, calculates the T-matrix, needed for the forces.
    '''
    nq = len(q)
    nj, modi_config = np.shape(config1.descriptors)
    _, dim = np.shape(config1.positions)
    nani, modi_desc = np.shape(descriptors_array)

    if not nq == modi_config == modi_desc:
        raise ValueError('The nr of q\'s does not match')

    dist = config1.nndistances
    # R_over_r.shape = (nj', ni', dim)
    R_over_r = config1.nndisplace_norm

    rq = dist.reshape(nj, nj, 1) * q.reshape(1, 1, nq)
    # cosrq.shape = (nj', ni', nq)
    cosrq = np.cos(rq)

    # cosrq_R.shape = (nj, dim, nq)
    cosrq_R = np.sum(cosrq.reshape(nj, nj, 1, nq) * R_over_r.reshape(nj, nj, dim, 1), axis=1)

    q2 = -2 * q

    # q2c.shape = (nani, nq)
    q2c = descriptors_array * q2

    return (cosrq_R @ q2c.T).reshape(nj * dim, nani)


# builds part of the row for the force kernel matrix given a configuration and a set of descriptors
def gaussian_force_mat(q: np.array, config1: configuration, descriptors_array: np.array, sigma: float) -> np.array:
    nq = len(q)
    nj, modi_config = np.shape(config1.descriptors)
    _, dim = np.shape(config1.positions)
    nani, modi_desc = np.shape(descriptors_array)

    if not nq == modi_config == modi_desc:
        raise ValueError('The nr of q\'s does not match')

    dist = config1.nndistances
    # R_over_r.shape = (nj', ni', dim)
    R_over_r = config1.nndisplace_norm

    rq = dist.reshape(nj, nj, 1) * q.reshape(1, 1, nq)
    # cosrq.shape = (nj', ni', nq)
    cosrq = np.cos(rq)

    # kern.shape = (nj, nani)
    kern = gaussian_kernel(config1.descriptors, descriptors_array, sigma)

    q_sig = (-1/(sigma**2) * q)
    # q_sig_Cia.shape = (nani, nq)
    q_sig_Cia = q_sig.reshape(1, nq) * descriptors_array
    # q_sig_Cj1.shape = (nj, nq)
    q_sig_Cj1 = q_sig.reshape(1, nq) * config1.descriptors
    # cosrq_Ror.shape = (nj', ni', dim, nq)
    cosrq_Ror = cosrq.reshape(nj, nj, 1, nq) * R_over_r.reshape(nj, nj, dim, 1)
    # sumi_cosrq_Ror.shape = (nj', dim, nq)
    sumi_cosrq_Ror = np.sum(cosrq_Ror, axis=1)

    # M_C.shape = (nj', dim, nani)
    M_Ciai1 = np.sum(np.sum(cosrq_Ror.reshape(nj, nj, dim, 1, nq) * kern.reshape(1, nj, 1, nani, 1), axis=1) * q_sig_Cia.reshape(1, 1, nani, nq), axis=3)
    M_Ci1i1 = np.sum(((np.sum(cosrq * q_sig_Cj1.reshape(1, nj, nq), axis=2)).reshape(nj, nj, 1) * R_over_r).reshape(nj, nj, dim, 1) * kern.reshape(1, nj, 1, nani), axis=1)
    M_Ciaj1 = np.sum(sumi_cosrq_Ror.reshape(nj, dim, 1, nq) * q_sig_Cia.reshape(1, 1, nani, nq), axis=3) * kern.reshape(nj, 1, nani)
    M_Cj1j1 = (np.sum(sumi_cosrq_Ror * q_sig_Cj1.reshape(nj, 1, nq), axis=2)).reshape(nj, dim, 1) * kern.reshape(nj, 1, nani)

    submat = (M_Ciai1 - M_Ci1i1 + M_Ciaj1 - M_Cj1j1).reshape(nj*dim, nani)

    return submat

class Kernel:
    def __init__(self, mode, *args):
        if mode == 'linear':
            self.kernel_mat = linear_kernel
            self.force_submat = linear_force_submat
        elif mode == 'gaussian':
            if not args:
                raise ValueError('For the Gaussian Kernel a sigma has to be supplied')
            self.kernel_mat = lambda x, y: gaussian_kernel(x, y, args[0])
            self.force_submat = lambda x, y, z: gaussian_force_mat(x, y, z, args[0])
        else:
            raise ValueError(f'kernel {mode} is not supported')
