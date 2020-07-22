import json
from time import time
from re import search, IGNORECASE
from copy import deepcopy
from statistics import pvariance
import numpy as np
from configuration import Configuration
import kernel

# load the global parameters used in the machine-learning calibration -> globals object für mehr übersicht??? ===

m_Si = 28.085 * 1.66 * 10**(-27) # kg
kB = 1.38 * 10**(-33) # A^2 kg fs^-2 K^-1
eV = 1.602177 * 10**(-19) # J/eV

a = 10.546640000 # lattice constant in A
mass = m_Si # We have Silicon
mass_ev = mass * 10**(2*15-2*10) / eV # eV fs^2 A^-2

with open('user_config.json', 'r') as user_conf:
    u_conf = json.load(user_conf)

directory = u_conf['file_out']
q = np.arange(1, u_conf['nr_modi']+1) * np.pi / u_conf['cutoff']
C_cal = np.array(np.loadtxt(directory + '/calibration_C.out'), dtype=float)
w_cal = np.array(np.loadtxt(directory + '/calibration_w.out'), dtype=float)
E_ave, _ = np.array(np.loadtxt(directory + '/calibration_E.out'), dtype=float)
kern = kernel.Kernel(*u_conf['kernel'])

# predicts the forces using the available machine-learned calibration
def predict_forces(config: Configuration):
    config.init_nn(u_conf['cutoff'], u_conf['lattice_vectors'])
    config.init_descriptor(q)
    N_ion, dim = np.shape(config.positions)

    # build the linear system
    # The first argument is the current configuration, the second one is the reference configuration
    K = kern.kernel_mat(config.descriptors, C_cal)
    K = np.sum(K, axis=0)[np.newaxis, :]
    T = kern.force_submat(q, config, C_cal)

    # solve the linear system
    EF = np.matmul(np.append(K,T, axis=0),w_cal)

    config.energy = EF[0] + E_ave
    config.forces = EF[1:].reshape(N_ion, dim)

    return config


# Randomly determines the starting conditions (positions, velocities) for the molecular dynamics simulation
def data_input_rand(T):
    # [T] = K

    # random positions, uniform distribution within lattice
    u_conf['lattice_vectors'] = np.eye(3) * a
    positions = np.random.rand(64,3) * a

    # random velocities, gaussian distribution at temperature T (= boltzmann distribution for speed)
    sigma_xyz = np.sqrt(63/64 * kB/mass * T) # A/fs
    velocities = np.random.normal(0, sigma_xyz, (64,3))
    for i in range(0,3):
        velocities[:,i] = velocities[:,i] - np.mean(velocities[:,i])

    #print("E_kin should:", 3/2 * kB * T * 63)
    #print("E_kin is:", 1/2 * mass * np.sum(velocities**2))

    # create the configuration object from the data
    config_0 = Configuration(positions, None, None, None, None, None, velocities)
    config_0 = predict_forces(config_0)

    return config_0


# Reads the starting conditions (positions, velocities) for the molecular dynamics simulation from a CONTCAR file
def data_input_contcar():

    filename = "CONTCAR"
    if not search(r'contcar', filename, IGNORECASE):
        raise ValueError(f'no contcar file')

    with open(filename, 'r') as contcar_in:
        contcar_content = contcar_in.readlines()

    a = float(contcar_content[1]) # overwrite default lattice constant
    u_conf['lattice_vectors'] = np.eye(3) * a

    dim = len(contcar_content[2].split()) # dimensions in space (usu. 3)
    Ni = int(contcar_content[3+dim]) # number of ions

    positions = contcar_content[5+dim : 5+dim+Ni]
    velocities = contcar_content[6+dim+Ni : 6+dim+2*Ni]
    for i in range(len(positions)):
        positions[i] = positions[i].split()
        velocities[i] = velocities[i].split()
    positions = np.array(positions, dtype=float) * a
    velocities = np.array(velocities, dtype=float) # === Muss velocities eventuell auch mit a multipliziert werden? Denke nicht weil Größenordnung passt

    config_0 = Configuration(positions, None, None, None, None, None, velocities)
    config_0 = predict_forces(config_0)

    return config_0


# equilibrates the system for given temperature T
def equilibrate(config, T, dt=0.01, doprint=False):

    # number Velocity-Verlet steps to equilibrate (only in tenths)
    t_0 = time()

    steps = 100
    for i in range(steps//10):
        if doprint:
            print(f'Equilibration in progress: {(i*1000)//steps} %', end='\r')

        config = veloverlet_10(dt, config)

        sig_theo = np.sqrt(63/64 * kB/mass * T) # theoretical variance
        sig_real = np.sqrt(pvariance(config.velocities.flatten(), 0)) # sample variance with known mean 0
        factor = sig_theo/sig_real # Normal distribution scales as: sig*N(0,x) = N(0,sig*x)
        config.velocities = config.velocities.reshape(np.shape(config.positions)) * factor

    if doprint:
        print(f'Equilibration: finished after {time()-t_0:.3} s')

    return config


# perform 10 velocity-verlet time steps, then write the NN-distances into an already open (!) file
def veloverlet_10(dt, config0, nn_file=None):

    # perform 10 steps at once to minimize correlations (now for testing purposes only 1 step)
    for i in range(10):
        # Velocity-Verlet algorithm: positions
        pos1 = config0.positions + config0.velocities * dt + config0.forces/(2*mass_ev) * dt**2
        pos1 = (pos1 % a + a) % a # periodic boundary conditions
        # print(pos1)
        config1 = Configuration(pos1)
        # Machine learned: forces
        config1 = predict_forces(config1)
        # Velocity-Verlet algorithm: velocities
        config1.velocities = config0.velocities + (config0.forces + config1.forces)/(2*mass_ev) * dt

        #print(id(config0), id(config1))
        config0 = deepcopy(config1)
        #print(id(config0.forces), id(config1.forces))
        #print(config0.forces)

    if not(nn_file is None):
        # append NN distances < Rcut to nn_dist output-file
        for i in range(np.shape(config0.nndistances)[0]):
            for j in range(i+1, np.shape(config0.nndistances)[1]): # each distance will be written only once. this is ok, if scaling is done that way too
                if config0.nndistances[i,j] != 0:
                    nn_file.write(str(config0.nndistances[i,j]) + "\r\n")

    return config0


# Write the configuration's parameter to an already open (!) file
def veloverlet_write(config, i, vv_file):

    dim = np.shape(config.positions)[1]

    vv_file.write("***** Iteration block (" + str(i) + ") ***** \r\n" + "\r\n")

    vv_file.write("--------------- POSITIONS --------------- \r\n")
    vv_file.write(str(config.positions) + "\r\n" + "\r\n")
    vv_file.write("center of mass: " + "\r\n")
    for d in range(dim):
        vv_file.write("     " + str(np.mean(config.positions[:,d])))
    vv_file.write("\r\n" + "\r\n")

    vv_file.write("--------------- VELOCITIES --------------- \r\n")
    vv_file.write(str(config.velocities) + "\r\n" + "\r\n")
    vv_file.write("mean speed: " + "\r\n" + "     " + str(np.sqrt(np.mean(config.velocities**2)*dim)) + "\r\n" + "\r\n")

    vv_file.write("--------------- FORCES --------------- \r\n")
    vv_file.write(str(config.forces) + "\r\n" + "\r\n")
    vv_file.write("mean force: " + "\r\n" + "     " + str(np.sqrt(np.mean(config.forces**2)*dim)) + "\r\n" + "\r\n")

    return


def main(dt=0.01, steps=1000, doprint=False):

    # initialize the starting configuration

    ##### For random configuration: #####
    #T = 1450 # targeted temperature for random velocities and equilibration in Kelvin
    #config_0 = data_input_rand(T)
    #config_0 = equilibrate(config_0, T, dt, doprint)

    ##### Read in configuration: #####
    config_0 = data_input_contcar()

    # create (or overwrite if it exists) the nn_dist.out and vv.out file
    nn_file = open(directory + '/nn.out', 'w')
    vv_file = open(directory + '/vv.out', 'w')
    veloverlet_write(config_0, 0, vv_file)

    # do #steps timesteps of size dt, print results if wished
    t_0 = time()
    for i in range(steps//10):
        if doprint:
            print(f'Molecular dynamics in progress: {(i*1000)//steps} %', end='\r')
        config_0 = veloverlet_10(dt, config_0, nn_file)
        veloverlet_write(config_0, i+1, vv_file)

    if doprint:
        print(f'Molecular dynamics: finished after {time()-t_0:.3} s')

    # close the nn_dist.out file as the simulation is finished
    nn_file.close()
    vv_file.close()


if __name__ == '__main__':
    dt = 0.1
    steps = 1000
    doprint = True
    main(dt, steps, doprint)

