import json
from time import time
from copy import deepcopy
import numpy as np
from configuration import Configuration
import kernel

# functions:
# + data_input: positions, velocities, dann equilibration ausführen
# - equilibrate: xxx steps mit velocities scaling für bestimmte temperatur
# + predict_forces: forces aus den positions berechnen nach unserer machine-learning kalibration
# + veloverlet_10: 10 schritte machen mit velocity verlet algorithmus, danach die abstände in ein file schreiben
# + main: starting condition, 100 mal verlet10 ausführen, jedes mal printen wie weit er is wenn True



# load the global parameters used in the machine-learning calibration
with open('user_config.json', 'r') as user_conf:
    u_conf = json.load(user_conf)
    
directory = u_conf['file_out']
q = np.arange(1, u_conf['nr_modi']+1) * np.pi / u_conf['cutoff']
C_cal = np.array(np.loadtxt(directory + '/calibration_C.out'), dtype=float)
w_cal = np.array(np.loadtxt(directory + '/calibration_w.out'), dtype=float)
E_ave, _ = np.array(np.loadtxt(directory + '/calibration_E.out'), dtype=float)
kern = kernel.Kernel(*u_conf['kernel'])

m_Si = 28.1 * 1.66 * 10**(-27) # kg
kB = 1.38 * 10**(-33) # A^2 kg fs^-2 K^-1
a = 10.546640000 # lattice constant in A

mass = m_Si # We choose Silicium

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
    

# Determines the starting conditions (positions, velocities) for the molecular dynamics simulation
# This is just a placeholder function as of now, depending on how we will create input data (read a file?) in the end
def data_input(T):
    # [T] = K
    
    # random positions, uniform distribution within lattice
    u_conf['lattice_vectors'] = np.eye(3) * a
    positions = np.random.rand(64,3) * a
    
    # random velocities, gaussian distribution at temperature T (= boltzmann distribution for speed)
    sigma_xyz = np.sqrt(63/64 * kB/mass * T) # A/fs
    velocities = np.random.normal(0, sigma_xyz, (64,3))
    for i in range(0,3):
        velocities[:,i] = velocities[:,i] - np.sum(velocities[:,i])/64
    
    #print("E_kin should:", 3/2 * kB * T * 63)
    #print("E_kin is:", 1/2 * mass * np.sum(velocities**2))
    
    # create the configuration object from the data
    config_0 = Configuration(positions, None, None, None, None, None, velocities)
    config_0 = predict_forces(config_0)
    print(config_0.velocities)
    
    return config_0
    

# === at the moment a dummy function
def equilibrate(config, T):
    return config
    

# perform 10 velocity-verlet time steps, then write the NN-distances in a file
def veloverlet_10(dt, config0, nn_file):
    
    # perform 10 steps at once to minimize correlations (now for testing purposes only 1 step)
    for i in range(10):
        # Velocity-Verlet algorithm: positions
        pos1 = config0.positions + config0.velocities * dt + config0.forces/(2*mass) * dt**2
        pos1 = (pos1 % a + a) % a # periodic boundary conditions
        #print(pos1)
        config1 = Configuration(pos1)
        # Machine learned: forces
        config1 = predict_forces(config1)
        # Velocity-Verlet algorithm: velocities
        config1.velocities = config0.velocities + (config0.forces + config1.forces)/(2*mass) * dt
        
        #print(id(config0), id(config1))
        config0 = deepcopy(config1)
        #print(id(config0.forces), id(config1.forces))
    
    # append NN distances < Rcut to nn_dist output-file
    for i in range(np.shape(config0.nndistances)[0]):
        for j in range(i+1, np.shape(config0.nndistances)[1]): # each distance will be written only once. this is ok, if scaling is done that way too
            if config0.nndistances[i,j] != 0:
                nn_file.write(str(config0.nndistances[i,j]) + "\r\n")
    
    return config0
    

def main(dt=0.01, steps=1000, doprint=False):
    
    # initialize the starting configuration
    T = 3000 # targeted temperature for random velocities and equilibration in Kelvin
    config_0 = data_input(T)
    config_0 = equilibrate(config_0, T)
    
    # create (or overwrite if it exists) the nn_dist.out file
    nn_file = open(directory + '/nn_dist.out', 'w')
    
    # do #steps timesteps of size dt, print results if wished
    t_0 = time()
    for i in range(steps//10):
        if doprint:
            print(f'Molecular dynamics in progress: {(i*1000)//steps} %', end='\r')
        config_0 = veloverlet_10(dt, config_0, nn_file)
    if doprint:
        print(f'Molecular dynamics: finished after {time()-t_0:.3} s')
        
    # close the nn_dist.out file as the simulation is finished
    nn_file.close()
    

if __name__ == '__main__':
    dt = 0.00000000001
    steps = 10
    doprint = True
    main(dt, steps, doprint)
    
