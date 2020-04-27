import json
from time import time
from math import pi
import numpy as np
from outcar_parser import Parser
from configuration import Configuration
import kernel

def load_data(u_conf: dict) ->  (int, int, np.array, list):
    # load parser and save nr of ions and lattice vectors
    parser = Parser(u_conf['file_in'])
    lattice_vectors = parser.find_lattice_vectors()
    lat_consts = np.diag(lattice_vectors.dot(lattice_vectors.T))

    # check if lattice constant is bigger than 2 rcut
    if any(np.greater(2 * u_conf["cutoff"], lat_consts)):
        raise ValueError('Cutoff cannot be bigger than half the lattice constants')

    # build the configurations from the parser
    # IDEA: Build training set.
    configurations = [
        Configuration(position, energy, force) for (energy, position, force) in parser
        .build_configurations(u_conf['stepsize'])
    ]

    return (len(configurations), parser.find_ion_nr(), lattice_vectors, configurations)


def init_configurations(u_conf: dict, configurations: list, q: np.array, C: np.array) -> None:
    N_conf = u_conf['N_conf']
    # calculate the nearest neighbors and the descriptors
    t0 = time() 
    for (alpha, config) in enumerate(configurations):
        config.init_nn(u_conf['cutoff'], u_conf['lattice_vectors'])
        config.init_descriptor(q)
        C[alpha, :, :] = config.descriptors
        print(f'calculating NN and descriptors: {alpha+1}/{N_conf}', end='\r')
    print(f'calculating NN and descriptors: finished after {time()-t0:.3} s')


def build_linear(u_conf: dict,configurations, C: np.array, q) -> (np.array, np.array, np.array, np.array):
    kern = kernel.Kernel(*u_conf['kernel'])
    N_conf = u_conf['N_conf']
    N_ion = u_conf['N_ion']
    # will be the super vectors
    E = np.zeros(N_conf)
    # this holds the matrix-elements in the shape [sum_j K(C^beta_j, C^alpha_i)]^beta_(alpha, i)
    K = np.zeros((N_conf, N_conf * N_ion))
    # Holds forces flattened
    F = np.zeros(N_conf * N_ion * 3)
    T = np.zeros((N_conf * N_ion * 3, N_conf * N_ion))    

    # build the linear system
    descr = C.reshape(N_conf * N_ion, len(q))
    t0 = time()
    for alpha in range(N_conf):
        print(f'Building [E, F, T]: {alpha+1}/{N_conf}', end='\r')
        E[alpha] = configurations[alpha].energy
        F[alpha*N_ion*3: (alpha+1)*N_ion*3] = configurations[alpha].forces.flatten()
        T[alpha*N_ion*3:(alpha+1)*N_ion*3] = kern.force_mat(qs, configurations[alpha], descr)
    print(f'Building [E, F, T]: finished after {time()-t0:.3} s')

    print('Building K:', end='\r')
    K = kern.kernel(descr, descr)
    K = np.sum(
        K.reshape(N_conf, N_ion, N_conf * N_ion),
        axis=1
    )
    print(f'Building K: finished after {time()-t0:.3} s')
    return (E, F, K, T)

def ridge_regression(K, E, lamb):
    X = np.matmul(np.transpose(K),K)
    y = np.matmul(np.transpose(K),E)
    # (X+lamb*I) * w - y = 0
    N = np.shape(K)[1]
    # IDEA: sklearn ridge-regression has several solving algorithms, maybe faster.
    w = np.linalg.solve(X + lamb * np.eye(N), y) # faster than: w = np.matmul(np.linalg.inv(X + lamb * np.eye(N)) , y)
    return w

def main():
    # load the simulation parameters
    with open('user_config.json', 'r') as u_conf:
        user_config = json.load(u_conf)

    # make a lsit of the allowed qs
    qs = np.arange(1, user_config['nr_modi']+1) * pi / user_config['cutoff'] 
    
    # read in data and save parameters for calibration comparison
    (user_config['N_conf'], user_config['N_ion'], user_config['lattice_vectors'], configurations) = load_data(user_config)

    # All descriptors in compact super-matrix
    C = np.zeros([user_config['N_ion'], user_config['nr_modi']])
    # compute the nn and configurations and fill them in C
    init_configurations(user_config, configurations, qs, C)
    
    # build the linear system
    (E, F, K, T) = build_linear(user_config, configurations, C, qs)

    t0 = time()
    # calculate the weights using ridge regression
    # IDEA: get quality of the fit with the sklearn function
    print('Solving linear system: ', end='')
    w_E = ridge_regression(K, E, user_config['lambda'])
    w_F = ridge_regression(T, F, user_config['lambda'])
    print(f'finished after {time()-t0:.3} s')

    
    # save calibration (file content will be overwritten if file already exists)
    np.savetxt('calibration_w.out', (w_E, w_F))
    np.savetxt('calibration_C.out', np.reshape(C, (user_cofig['N_conf'], user_config['N_ion'] * user_config['nr_modi'])))
    # loading: C_cal = np.reshape(np.loadtxt('calibration.out'), (N_conf, N_ion, user_config['nr_modi']))
    
if __name__ == '__main__':
    main()
