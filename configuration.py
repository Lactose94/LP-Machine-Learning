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
    #magnitude = np.sqrt(dr.dot(dr))
    return dr

class Configuration(object):
    
    # Es wäre gut, wenn wir beim parser die Reihenfolge (positions, energy, forces) verwenden würden,
    # da wir nur bei der Kalibration alle 3 haben. Bei der Anwendung hätten wir nur mehr Positions
    # (daher energy und forces in der Klasse configurations mit None).
    def __init__(self, positions, energy=None, forces=None, nndisplacements=None, nndistances=None, descriptors=None):
        self.positions = positions
        self.energy = energy
        self.forces = forces
        self.nndisplacements = nndisplacements
        self.nndistances = nndistances
        self.descriptors = descriptors
        
    # Diese Funktion erstellt die nearest-neighbour-tables für die Positionen und die Abstände.
    # Dafür muss die float-Variable rcut in Angström übergeben werden.
    def init_nn(self, rcut, lattice):
        n = np.shape(self.positions)[0]
        self.nndisplacements = [[] for i in range(n)] # n lists of variable length inside a list
        self.nndistances = [[] for i in range(n)] # n lists of variable length inside a list
        
        # get a vector of all lattice constants (primitive orthorhombic or cubic cell)
        a = lattice.diagonal()
            
        for i in range(n): # loop over central atoms
            for j in range(i+1,n): # loop over possible nearest neighbours
                rj_ri = dist(self.positions[i,:], self.positions[j,:], a)
                dr = np.sqrt(rj_ri.dot(rj_ri))
                if dr < rcut:
                    self.nndisplacements[i].append(rj_ri) # NN atom - central atom
                    self.nndisplacements[j].append(-rj_ri) # NN atom - central atom
                    self.nndistances[i].append(dr)
                    self.nndistances[j].append(dr)
        
    # Diese Funktion erstellt die descriptor coefficients der configuration.
    # Dafür muss ein float-Vektor q übergeben werden.
    # Dass dieser mit rcut zusammenpasst wird vorausgesetzt und nicht weiter überprüft.
    def init_descriptor(self, q):
        if self.nndistances is None or self.nndisplacements is None:
            print("Execute Configuration.init_nn(rcut,lattice) before calculating descriptor coefficients!")
            return
        else:
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
    print(config1.nndisplacements)
    print(config1.nndistances)
    
    # test init_descriptor
    config1.init_descriptor(q)
    print(config1.descriptors)
