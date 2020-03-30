# Klasse dessen Instanzen je eine Ionen-Konfiguration darstellen.
# In dieser Konfiguration ist ihre Energie, sowie die Positionen, und Kräfte der einzelnen Ionen hinterlegt.
# Durch die klasseneigene Methode init_nn können unter Angabe eines cutoff radius rcut ein nearest-neighbour-table
# für sowohl die Indizes der NN als auch die Abstände zu diesen erstellt werden.
# Durch die klasseneigene Methode init_descriptor können unter Angabe eines q-Vektors (dieser bestimmt die
# Koeffizienten in den Basis-Sinusfunktionen) danach die descriptor coefficients erstellt werden.

class Configuration(object):
    
    # Es wäre gut, wenn wir beim parser die Reihenfolge (positions, energy, forces) verwenden würden,
    # da wir nur bei der Kalibration alle 3 haben. Bei der Anwendung hätten wir nur mehr Positions
    # (daher energy und forces in der Klasse configurations mit None).
    def __init__(self, positions, energy=None, forces=None, nnindices = None, nndistances = None, descriptors = None):
        self.positions = positions
        self.energy = energy
        self.forces = forces
        self.nnindices = nnindices
        self.nndistances = nndistances
        self.descriptors = descriptors
        
        # Diese Funktion erstellt die nearest-neighbour-tables für die Indizes und die Abstände.
        # Dafür muss die float-Variable rcut in Angström übergeben werden.
    def init_nn(self, rcut):
        pass
        
    # Diese Funktion erstellt die descriptor coefficients der configuration.
    # Dafür muss ein float-Vektor q übergeben werden.
    # Dass dieser mit rcut zusammenpasst wird vorausgesetzt und nicht weiter überprüft.
    def init_descriptor(self, q):
        if self.nndistances is None or self.nnindices is None:
            print("Execute Configuration.init_nn(rcut) before calculating descriptor coefficients!")
            return
        pass
            
if __name__ == '__main__':
    config1 = Configuration([0,0,0])
    config1.init_nn(4)
    config1.init_descriptor([1,2,4])
