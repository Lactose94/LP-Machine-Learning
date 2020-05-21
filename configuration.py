# Klasse dessen Instanzen je eine Ionen-Konfiguration darstellen.
# In dieser Konfiguration ist ihre Energie, sowie die Positionen, und Kräfte der einzelnen Ionen hinterlegt.
# Durch die klasseneigene Methode init_nn können unter Angabe eines cutoff radius rcut ein nearest-neighbour-table
# für sowohl die Positionen der NN als auch die Abstände zu diesen erstellt werden.
# Durch die klasseneigene Methode init_descriptor können unter Angabe eines q-Vektors (dieser bestimmt die
# Koeffizienten in den Basis-Sinusfunktionen) danach die descriptor coefficients erstellt werden.

import numpy as np

def dist(r1, r2, a=1):
    dr = r2 - r1
    dr = dr - a * np.rint(dr/a) # rint = rounding to nearest integer (up or down)
    return dr

class Configuration(object):
    
    def __init__(self, positions, energy=None, forces=None, nndisplace_norm=None, nndistances=None, descriptors=None):
        self.positions = positions
        self.energy = energy
        self.forces = forces
        self.nndisplace_norm = nndisplace_norm # distances (vector) NORMALIZED (!), with 0 if not NN or self
        self.nndistances = nndistances # distances (scalar) with 0 if not NN or self
        self.descriptors = descriptors
        
    # Diese Funktion erstellt die nearest-neighbour-tables für die Positionen und die Abstände.
    # Dafür muss die float-Variable rcut in Angström übergeben werden.
    def init_nn(self, rcut, lattice):
        n, dim = np.shape(self.positions) # nr of atoms, nr of dimensions
        # n x n x dim numpy.array of normalized NN-displacement table
        self.nndisplace_norm = np.zeros((n, n, dim)) # 0 if self atom or not NN
        # n x n numpy.array of NN-distances table
        self.nndistances = np.zeros((n, n)) # 0 if self atom or not NN
        
        # get a vector of all lattice constants (primitive orthorhombic or cubic cell)
        a = lattice.diagonal()
            
        # TODO: Diese Loops kann man eventuell auch wegrationalisieren!
        for i in range(n): # loop over central atoms
            for j in range(i+1,n): # loop over possible nearest neighbours
                rj_ri = dist(self.positions[i,:], self.positions[j,:], a)
                dr = np.sqrt(rj_ri.dot(rj_ri))
                if dr < rcut:
                    self.nndisplace_norm[i,j,:] = rj_ri / dr # NN atom - central atom
                    self.nndisplace_norm[j,i,:] = - rj_ri / dr # NN atom - central atom
                    self.nndistances[i,j] = dr
                    self.nndistances[j,i] = dr
    
    # Diese Funktion erstellt die descriptor coefficients der configuration.
    # Dafür muss ein float-Vektor q übergeben werden.
    # Dass dieser mit rcut zusammenpasst wird vorausgesetzt und nicht weiter überprüft.
    def init_descriptor(self, q):
        if self.nndistances is None:
            print("Execute Configuration.init_nn(rcut,lattice) before calculating descriptor coefficients!")
            return
        else:
            # n x len(q) numpy.array of Descriptor coefficients
            self.descriptors = np.sum(np.sin(np.multiply.outer(self.nndistances, q)), axis=1)
        return
    
if __name__ == '__main__':
    
    rcut = 4
    q = np.array([np.pi/rcut,2*np.pi/rcut,3*np.pi/rcut])
    lattice = np.array([[10.0 ,  0.0],
                        [0.0 , 10.0]])
    positions = np.array([[1.0 , 1.0], # hat 3 NN
                          [1.0 , 9.0], # hat 2 NN
                          [3.0 , 3.0], # hat 1 NN
                          [9.0 , 9.0]]) # hat 2 NN
    
    # test __init__
    config1 = Configuration(positions)
    
    # test init_nn
    config1.init_nn(rcut, lattice)
    print(config1.nndisplace_norm)
    print(config1.nndistances)
    print(type(config1.nndistances[0]))
    
    # test init_descriptor
    config1.init_descriptor(q)
    print(config1.descriptors)
