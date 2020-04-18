import json
from time import time
from math import pi
import numpy as np
from outcar_parser import Parser
from configuration import Configuration
import kernel

def ridge_regression(K, E, lamb):
    X = np.matmul(np.transpose(K),K)
    y = np.matmul(np.transpose(K),E)
    # X*w - y + lamb*w = 0
    N = np.shape(K)[1]
    w = np.matmul(np.linalg.inv(X + lamb * np.eye(N)) , y)
    return w

def main():
    # load the simulation parameters
    with open('user_config.json', 'r') as u_conf:
        user_config = json.load(u_conf)

    # make a lsit of the allowed qs
    qs = np.arange(1, user_config['nr_modi']+1) * pi / user_config['cutoff'] 
    
    # choose kernel
    kern = kernel.Kernel(*user_config['kernel'])

    # load parser and save nr of ions and lattice vectors
    parser = Parser(user_config['file_in'])
    N_ion = parser.find_ion_nr()
    user_config['ion_nr'] = N_ion
    lattice_vectors = parser.find_lattice_vectors()
    user_config['lattice_vectors'] = lattice_vectors

    # check if lattice constant is bigger than 2 rcut
    lat_consts = np.diag(lattice_vectors.dot(lattice_vectors.T))
    if any(np.greater(2 * user_config['cutoff'], lat_consts)):
        raise ValueError('Cutoff cannot be bigger than half the lattice constants')

    # build the configurations from the parser
    configurations = [
        Configuration(position, energy, force) for (energy, position, force) in parser
        .build_configurations(user_config['stepsize'])
    ]

    N_conf = len(configurations)

    # calculate the nearest neighbors and the descriptors
    print('calculating NN and descriptors')
    compl_descriptors = np.zeros((N_conf, N_ion, len(qs)))

    t0 = time()
    for (alpha, config) in enumerate(configurations):
        config.init_nn(user_config['cutoff'], lattice_vectors)
        config.init_descriptor(qs)
        compl_descriptors[alpha] = config.descriptors

        print(f'{alpha}/{N_conf}', end='\r')
    t1 = time()
    print(f'finished after {t1 - t0} s')
    # will be the super vectors
    E = np.zeros(N_conf)
    # this holds the matrix-elements in the shape [sum_j K(C^beta_j, C^alpha_i)]^beta_(alpha, i)
    K = np.zeros((N_conf, N_conf * N_ion))
    # Holds forces flattened
    F = np.zeros(N_conf * N_ion * 3)
    T = np.zeros((N_conf * N_ion * 3, N_conf * N_ion))    

    # build the linear system
    descr = compl_descriptors.reshape(N_conf * N_ion, len(qs))
    t0 = time()
    print('Building linear system')
    for alpha in range(N_conf):
        print(f'{alpha+1}/{N_conf}', end='\r')
        E[alpha] = configurations[alpha].energy
        F[alpha*N_ion*3: (alpha+1)*N_ion*3] = configurations[alpha].forces.flatten()
        T[alpha*N_ion*3:(alpha+1)*N_ion*3] = kern.force_mat(qs, configurations[alpha], descr)

    K = kern.kernel(descr, descr)
    K = np.sum(
        K.reshape(N_conf, N_ion, N_conf * N_ion),
        axis=1
    )
    t1 = time()
    print(f'finished after {t1 - t0} s')
    
    # calculate the weights using ridge regression
    t0 = time()
    print('Solving linear system')
    w_E = ridge_regression(K, E, user_config['lambda'])
    w_F = ridge_regression(T, F, user_config['lambda'])
    t1 = time()
    print(f'finished after {t1 - t0} s')
    
    # save calibration (file content will be overwritten if it already exists)
    np.savetxt('calibration.out', (w_E, w_F))
    
if __name__ == '__main__':
    main()
