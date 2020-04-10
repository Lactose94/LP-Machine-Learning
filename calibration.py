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
    user_config['q'] = np.array(list(map(
        lambda n: n * pi / user_config['cutoff'],
        range(1, user_config['nr_modi"']+1)
    )))


    
    # check if the kernel is implemented
    mode = user_config['kernel']
    if mode not in MODI:
        raise ValueError('The choosen Kernel "{mode}" is not implemented')


    # load parser and save nr of ions and lattice vectors
    parser = Parser(user_config['file_in'])
    user_config['ion_nr'] = parser.find_ion_nr()
    user_config['lattice_vectors'] = parser.find_lattice_vectors()

    # TODO: check if lattice constant is bigger than 2 rcut
    lat_consts = np.array(np.linalg.norm(vec) for vec in user_config['lattice_vectors'])
    if any(2 * user_config['cutoff'] > lat_consts):
        raise ValueError('Cutoff cannot be bigger than half the lattice constants')
    
    # build the configurations from the parser
    configurations = [
        Configuration(position, energy, force) for (energy, position, force) in parser
        .build_configurations(user_config['stepsize'])
    ]

    # calculate the nearest neighbors and the descriptors
    for config in configurations:
        config.init_nn(user_config['cutoff'])
        config.init_descriptor(user_config['q'])

if __name__ == '__main__':
    main()
