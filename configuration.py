# Klasse dessen Instanzen je eine Ionen-Konfiguration darstellen.
# In dieser Konfiguration ist ihre Energie, sowie die Positionen, und Kräfte der einzelnen Ionen hinterlegt.
# Durch die klasseneigene Methode init_NN können unter Angabe eines cutoff radius rcut ein Nearest-Neighbour-table
# für sowohl die Indices der NN als auch die Abstände zu diesen erstellt werden.
# Durch die klasseneigene Methode init_descriptor können unter Angabe eines q-Vektors (dieser bestimmt die
# Koeffizienten in den Basis-Sinusfunktionen) danach die descriptor coefficients erstellt werden.

class Configuration(object):
    
    def __init__(self, positions, energy=None, forces=None, NNindices = None, NNdistances = None, descriptors = None):
        self.positions = positions
        self.energy = energy
        self.forces = forces
        self.NNindices = None
        self.NNdistances = None
        self.descriptors = None
        
        # Diese Funktion erstellt die Nearest Neighbour Tables für die Indizes und die Abstände.
        # Dafür muss die float-Variable rcut in Angström übergeben werden.
        def init_NN(self, rcut):
            pass
        
        # Diese Funktion erstellt die descriptor coefficients der configuration.
        # Dafür muss ein float-Vektor q übergeben werden.
        # Dass dieser mit rcut zusammenpasst wird vorausgesetzt und nicht weiter überprüft.
        def init_descriptor(self, q):
            if self.NNdistances is None or self.NNindices is None:
                print("Execute Configuration.init_NN(rcut) before calculating descriptor coefficients!")
                return
            pass