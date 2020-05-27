import json
from time import time
from math import pi
import numpy as np
from outcar_parser import Parser
from configuration import Configuration
import kernel


def load_data(u_conf: dict) ->  (int, int, np.array, list):
    '''
    Loads the data from the file specified in u_conf and returns the parameters of the
    simulation as (N_conf, N_ion, lattice vectors, list of configurations)
    '''
    
    stepsize = 1000
    
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
        Configuration(position, energy, forces) for (energy, position, forces) in parser
        .build_configurations(stepsize)
    ]

    return (len(configurations), parser.find_ion_nr(), lattice_vectors, configurations)


def init_configurations(u_conf: dict, configurations: list, q: np.array, C: np.array) -> None:
    '''
    Initializes the nearest neighbors and descriptors. Writes values into the C array.
    Choosen this way, to only have sideeffects and no return.
    '''
    n_conf = u_conf['N_conf']
    # calculate the nearest neighbors and the descriptors
    t_0 = time()
    for (alpha, config) in enumerate(configurations):
        config.init_nn(u_conf['cutoff'], u_conf['lattice_vectors'])
        config.init_descriptor(q)
        C[alpha, :, :] = config.descriptors
        print(f'calculating NN and descriptors: {alpha+1}/{n_conf}', end='\r')
    print(f'calculating NN and descriptors: finished after {time()-t_0:.3} s')


def predict_linear(u_conf: dict, configurations, C: np.array, q) -> (np.array, np.array, np.array, np.array):
    '''
    Loads the calibration data, ntializes the kernel and then builds the linear system with the kernel matrices according to kernel
    '''
    C_cal = np.array(np.loadtxt('calibration_C.out'), dtype=float)
    
    kern = kernel.Kernel(*u_conf['kernel'])
    n_conf = u_conf['N_conf']
    n_ion = u_conf['N_ion']
    nc_ni = np.shape(C_cal)[0]
    # will be the super vectors
    E = np.zeros(n_conf)
    # Holds forces flattened
    F = np.zeros(n_conf * n_ion * 3)
    T = np.zeros((n_conf * n_ion * 3, nc_ni))
    
    # build the linear system
    
    t_0 = time()
    print('Building K ...', end='\r')
    descr_new = C.reshape(n_conf * n_ion, len(q))
    # das erste Argument ist die aktuelle Konfiguration, das zweite die Referenz-Konfiguration
    # this holds the matrix-elements in the shape [sum_j K(C^beta_j, C^alpha_i)]^beta_(alpha, i)
    K = kern.kernel_mat(descr_new, C_cal)
    K = np.sum(
        K.reshape(n_conf, n_ion, nc_ni),
        axis=1
    )
    print(f'Building K: finished after {time()-t_0:.3} s')
    
    t_0 = time()
    for alpha in range(n_conf):
        print(f'Building T: {alpha+1}/{n_conf}', end='\r')
        E[alpha] = configurations[alpha].energy
        F[alpha*n_ion*3: (alpha+1)*n_ion*3] = configurations[alpha].forces.flatten()
        T[alpha*n_ion*3:(alpha+1)*n_ion*3] = kern.force_submat(q, configurations[alpha], C_cal)
    print(f'Building T: finished after {time()-t_0:.3} s')
    
    return (E, F, K, T)


def ridge_prediction(K, w):
    E = np.matmul(K, w)
    return E


def main():
    # load the simulation parameters
    with open('user_config.json', 'r') as u_conf:
        user_config = json.load(u_conf)

    # make a list of the allowed qs
    qs = np.arange(1, user_config['nr_modi']+1) * pi / user_config['cutoff']

    # read in data and save parameters for calibration comparison
    (user_config['N_conf'], user_config['N_ion'], user_config['lattice_vectors'], configurations) = load_data(user_config)

    # All descriptors in compact super-matrix
    C = np.zeros([user_config['N_conf'], user_config['N_ion'], user_config['nr_modi']])
    # compute the nn and configurations and fill them in C
    init_configurations(user_config, configurations, qs, C)

    # build the linear system
    (E, F, K, T) = predict_linear(user_config, configurations, C, qs)
    
    w_cal = np.array(np.loadtxt('calibration_w.out'), dtype=float)
    E_ave, _ = np.array(np.loadtxt('calibration_E.out'), dtype=float)
    
    t_0 = time()
    # calculate the weights using ridge regression
    # IDEA: get quality of the fit with the sklearn function
    print('Solving linear system ... ', end='\r')
    EF = ridge_prediction(np.append(K,T, axis=0),w_cal)
    print(f'Solving linear system: finished after {time()-t_0:.3} s')
    
    E_cal = EF[0:user_config['N_conf']] + E_ave
    E_msd = np.mean((E - E_cal)**2)
    F_cal = EF[user_config['N_conf']:] # .reshape((user_config['N_conf'], user_config['N_ion'], 3))
    F_msd = np.mean((F - F_cal)**2)
    
    # show results in compact way
    print("calculated Energies have a variance / std-deviation of:")
    print('{:.5f}'.format(E_msd), "/", '{:.5f}'.format(np.sqrt(E_msd)))
#    print("Positions:")
#    print([conf.positions for conf in configurations])
    print("calculated Forces have a variance / std-deviation of:")
    print('{:.5f}'.format(F_msd), "/", '{:.5f}'.format(np.sqrt(F_msd)))
    show = input("Show energies and forces? y:yes, n:no -> ")
    if show == "y":
        print("original Energies:")
        print(E)
        print("calculated Energies:")
        print(E_cal)
        print("original Forces:")
        print(F.reshape((user_config['N_conf'], user_config['N_ion'], 3)))
        print("calculated Forces:")
        print(F_cal.reshape((user_config['N_conf'], user_config['N_ion'], 3)))
    
if __name__ == '__main__':
    main()
