import json
from math import pi
import numpy as np
from outcar_parser import Parser
from configuration import Configuration
import kernel

def main():
    # load the simulation parameters
    with open('user_config.json', 'r') as u_conf:
        user_config = json.load(u_conf)

    # make a lsit of the allowed qs
    qs = np.array(list(map(
        lambda n: n * pi / user_config['cutoff'], range(1, user_config['nr_modi"']+1)
    )))

    # choose kernel
    used_kernel = kernel.Kernel(user_config['kernel'])

    # load parser and save nr of ions and lattice vectors
    parser = Parser(user_config['file_in'])
    Nion = parser.find_ion_nr()
    user_config['ion_nr'] = Nion
    lattice_vectors = parser.find_lattice_vectors()
    user_config['lattice_vectors'] = lattice_vectors

    # check if lattice constant is bigger than 2 rcut
    lat_consts = np.array(np.linalg.norm(vec) for vec in lattice_vectors)
    if any(2 * user_config['cutoff'] > lat_consts):
        raise ValueError('Cutoff cannot be bigger than half the lattice constants')
    
    # build the configurations from the parser
    configurations = [
        Configuration(position, energy, force) for (energy, position, force) in parser
        .build_configurations(user_config['stepsize'])
    ]

    Nconf = len(configurations)

    # calculate the nearest neighbors and the descriptors
    for config in configurations:
        config.init_nn(user_config['cutoff'], lattice_vectors)
        config.init_descriptor(qs)

    # will be the super vectors
    E = np.zeros(Nconf)
    # this holds the matrix-elements in the shape [sum_j K(C^beta_j, C^alpha_i)]^beta_(alpha, i)
    K = np.zeros((Nconf, Nconf * Nion))

    # build the linear system
    # TODO: Also calculate forces
    for alpha in range(Nconf):
        E[alpha] = configurations[alpha].energy
        for beta in range(Nconf):
            K[alpha, beta: beta + Nion] = used_kernel.build_subrow(configurations[alpha], configurations[beta])


if __name__ == '__main__':
    main()
