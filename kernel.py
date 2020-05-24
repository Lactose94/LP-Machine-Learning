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

"""
def derivatives(q: np.array, displ: np.array, dist: np.array) -> np:
    '''
    Calculate the derivatives matrix of the descriptors
    '''
    nq = len(q)
    nc, ni, _, d = displ.shape
    rq = dist.reshape(nc, ni, ni, 1) * q
    qcosrq = q * np.cos(rq)
    R_over_r = displ / dist.reshape(nc, ni, ni, 1)
    D = qcosrq.reshape(nc, ni, ni, 1, nq) *  R_over_r.reshape(nc, ni, ni, d, 1)

    return D
"""

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
    
    rq = dist.reshape(nj,nj,1) * q.reshape(1,1,nq)
    # cosrq.shape = (nj', ni', nq)
    cosrq = np.cos(rq)
    
    # kern.shape = (nj, nani)
    kern = gaussian_kernel(config1.descriptors, descriptors_array, sigma)
    
    q_sig = (-1/(sigma**2) * q)
    # q_sig_Cia.shape = (nani, nq)
    q_sig_Cia = q_sig.reshape(1,nq) * descriptors_array
    # q_sig_Cj1.shape = (nj, nq)
    q_sig_Cj1 = q_sig.reshape(1,nq) * config1.descriptors
    # cosrq_Ror.shape = (nj', ni', dim, nq)
    cosrq_Ror = cosrq.reshape(nj,nj,1,nq) * R_over_r.reshape(nj,nj,dim,1)
    # sumi_cosrq_Ror.shape = (nj', dim, nq)
    sumi_cosrq_Ror = np.sum(cosrq_Ror, axis=1)
    
    # M_C.shape = (nj', dim, nani)
    M_Ciai1 = np.sum(np.sum(cosrq_Ror.reshape(nj,nj,dim,1,nq) * kern.reshape(1,nj,1,nani,1), axis=1) * q_sig_Cia.reshape(1,1,nani,nq), axis=3)
    M_Ci1i1 = np.sum(((np.sum(cosrq * q_sig_Cj1.reshape(1,nj,nq), axis=2)).reshape(nj,nj,1) * R_over_r).reshape(nj,nj,dim,1) * kern.reshape(1,nj,1,nani), axis=1)
    M_Ciaj1 = np.sum(sumi_cosrq_Ror.reshape(nj,dim,1,nq) * q_sig_Cia.reshape(1,1,nani,nq), axis=3) * kern.reshape(nj,1,nani)
    M_Cj1j1 = (np.sum(sumi_cosrq_Ror * q_sig_Cj1.reshape(nj,1,nq), axis=2)).reshape(nj,dim,1) * kern.reshape(nj,1,nani)
    
    submat = (M_Ciai1 - M_Ci1i1 + M_Ciaj1 - M_Cj1j1).reshape(nj*dim,nani)
    
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
