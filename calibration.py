import json
from math import pi, exp
import numpy as np
from outcar_parser import Parser
from configuration import Configuration


def linear_kernel(descriptor1: array, descriptor2: array) -> float:
    if np.shape(descriptor1) != np.shape(descriptor2):
        raise ValueError('Shapes of input do not match')

    return np.inner(descriptor1, descriptor2)


def gaussian_kernel(descriptor1: array, descriptor2: array, sigma: float) -> float:
    if np.shape(descriptor1) != np.shape(descriptor2):
        raise ValueError('Shapes of input do not match')

    return exp(np.linalg.norm(descriptor1 - descriptor2)**2 / (2 * sigma**2))

MODI = {
    'linear': linear_kernel,
    'gaussian': gaussian_kernel
    }


if __name__ == '__main__':
    # load the simulation parameters
    with open('user_config.json', 'r') as u_conf:
        user_config = json.load(u_conf)

    # make a lsit of the allowed qs
    user_config['q'] = list(map(
        lambda n: n * pi / user_config['cutoff'],
        range(1, user_config['nr_modi"']+1)
        ))

    # load parser and save nr of ions and lattice vectors
    parser = Parser(user_config['file_in'])
    user_config['ion_nr'] = parser.find_ion_nr()
    user_config['lattice_vectors'] = parser.find_lattice_vectors()

    # build the configurations from the parser
    configurations = [
        Configuration(position, energy, force) for (energy, position, force) in parser
            .build_configurations(user_config['stepsize'])
    ]

    # calculate the nearest neighbors and the descriptors
    for config in configurations:
        config.init_nn(user_config['cutoff'])
        config.init_descriptor(user_config['q'])