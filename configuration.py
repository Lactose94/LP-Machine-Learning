# Klasse dessen Instanzen je eine Ionen-Konfiguration darstellen.
# In dieser Konfiguration ist ihre Energie, sowie die Positionen, und Kräfte der einzelnen Ionen hinterlegt.
# Durch die klasseneigene Methode init_nn können unter Angabe eines cutoff radius rcut ein nearest-neighbour-table
# für sowohl die Positionen der NN als auch die Abstände zu diesen erstellt werden.
# Durch die klasseneigene Methode init_descriptor können unter Angabe eines q-Vektors (dieser bestimmt die
# Koeffizienten in den Basis-Sinusfunktionen) danach die descriptor coefficients erstellt werden.

import numpy as np
from math import sqrt, sin

class Configuration(object):
    
    # Es wäre gut, wenn wir beim parser die Reihenfolge (positions, energy, forces) verwenden würden,
    # da wir nur bei der Kalibration alle 3 haben. Bei der Anwendung hätten wir nur mehr Positions
    # (daher energy und forces in der Klasse configurations mit None).
    def __init__(self, positions, energy=None, forces=None, nnpositions=None, nndistances=None, descriptors=None):
        self.positions = positions
        self.energy = energy
        self.forces = forces
        self.nnpositions = nnpositions
        self.nndistances = nndistances
        self.descriptors = descriptors
        
    # Diese Funktion erstellt die nearest-neighbour-tables für die Positionen und die Abstände.
    # Dafür muss die float-Variable rcut in Angström übergeben werden.
    def init_nn(self, rcut):
        pass
        
    # Diese Funktion erstellt die descriptor coefficients der configuration.
    # Dafür muss ein float-Vektor q übergeben werden.
    # Dass dieser mit rcut zusammenpasst wird vorausgesetzt und nicht weiter überprüft.
    def init_descriptor(self, q):
        if self.nndistances is None or self.nnpositions is None:
            print("Execute Configuration.init_nn(rcut) before calculating descriptor coefficients!")
            return
        if self.descriptors is None:
            m = np.shape(self.positions)[0]
            n = np.size(q)
            self.descriptors = np.zeros((m, n))
            for i in range(0,m): # loop over central atoms
                nrnn = np.size(self.nndistances[i]) # number of nearest neighbours for atom i
                for j in range(0,n): # loop over q
                    for k in range(0,nrnn): # loop over nearest neighbours of atom i
                        self.descriptors[i,j] += sin(q[j] * self.nndistances[i][k])
        return
            
if __name__ == '__main__':
    
    q = [4,2,1]
    rcut = 2
    
    # test __init__
    positions = np.array([[1 , 1], # hat keine NN
                          [7 , 9], # hat 1 NN
                          [8 , 8], # hat 2 NN
                          [9 , 7]]) # hat 1 NN
    config1 = Configuration(positions)
    config1.init_nn(rcut)
    config1.init_descriptor(q)
    
    # test init_nn
    
    # test init_descriptor
    nndistances = [[], 
                   [sqrt(2)], 
                   [sqrt(2) , sqrt(2)], 
                   [sqrt(2)]]
    nnpositions = [[], 
                   [[8 , 8]], 
                   [[7 , 9] , [9 , 7]], 
                   [[8 , 8]]]
    config3 = Configuration(positions, None, None, nnpositions, nndistances, None)
    config3.init_descriptor(q)
    print(config3.descriptors)
