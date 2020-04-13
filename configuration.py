# Klasse dessen Instanzen je eine Ionen-Konfiguration darstellen.
# In dieser Konfiguration ist ihre Energie, sowie die Positionen, und Kräfte der einzelnen Ionen hinterlegt.
# Durch die klasseneigene Methode init_nn können unter Angabe eines cutoff radius rcut ein nearest-neighbour-table
# für sowohl die Positionen der NN als auch die Abstände zu diesen erstellt werden.
# Durch die klasseneigene Methode init_descriptor können unter Angabe eines q-Vektors (dieser bestimmt die
# Koeffizienten in den Basis-Sinusfunktionen) danach die descriptor coefficients erstellt werden.

import numpy as np

def difference(r1, r2, a=1):
    dr = r2 - r1
    dr = dr - a * np.rint(dr/a) # rint = rounding to nearest integer (up or down)
    return dr

class Configuration(object):
    
    # Es wäre gut, wenn wir beim parser die Reihenfolge (positions, energy, forces) verwenden würden,
    # da wir nur bei der Kalibration alle 3 haben. Bei der Anwendung hätten wir nur mehr Positions
    # (daher energy und forces in der Klasse configurations mit None).
    def __init__(self, positions, energy=None, forces=None, differences=None, distances=None, NNlist=None, descriptors=None):
        self.positions = positions
        self.energy = energy
        self.forces = forces
        self.differences = differences
        self.distances = distances
        self.NNlist = NNlist
        self.descriptors = descriptors
        
    # Diese Funktion erstellt die nearest-neighbour-tables für die Positionen und die Abstände.
    # Dafür muss die float-Variable rcut in Angström übergeben werden.
    def init_nn(self, rcut, lattice):
        Nions, _ = np.shape(self.positions)
        self.distances = np.zeros((Nions, Nions, 3)) 
        self.distances = np.zeros((Nions, Nions)) 
        
        # get a vector of all lattice constants (primitive orthorhombic or cubic cell)
        a = lattice.diagonal()
            
        for i in range(Nions): # loop over central atoms
            for j in range(i+1,Nions): # loop over possible nearest neighbours
                dR = difference(self.positions[i,:], self.positions[j,:], a)
                self.differences[i, j] = dR
                self.distances[i, j] = np.sqrt(dR.dot(dR))

                self.NNlist = np.nonzero(self.distances < rcut )
        
    # Diese Funktion erstellt die descriptor coefficients der configuration.
    # Dafür muss ein float-Vektor q übergeben werden.
    # Dass dieser mit rcut zusammenpasst wird vorausgesetzt und nicht weiter überprüft.
    def init_descriptor(self, q):
        if self.distances is None or self.differences is None:
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
                        self.descriptors[i,j] += np.sin(q[j] * self.nndistances[i][k])
        return
            
if __name__ == '__main__':
    
    rcut = 4
    q = [np.pi/rcut,2*np.pi/rcut,3*np.pi/rcut]
    lattice = np.np.array([[10.0 ,  0.0],
                        [0.0 , 10.0]])
    positions = np.np.array([[1.0 , 1.0], # hat 3 NN
                          [1.0 , 9.0], # hat 2 NN
                          [3.0 , 3.0], # hat 1 NN
                          [9.0 , 9.0]]) # hat 2 NN
    
    # test __init__
    config1 = Configuration(positions)
    
    # test init_nn
    config1.init_nn(rcut, lattice)
    print(config1.nnpositions)
    print(config1.nndistances)
    
    # test init_descriptor
    config1.init_descriptor(q)
    print(config1.descriptors)
