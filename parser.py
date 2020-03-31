from re import match, search, IGNORECASE
from numpy import array, shape, fromstring, array_equal, eye, inner


# Hier werden die patterns und Textbausteine gespeichert, nach denen später das File durchsucht oder aufgespalten wird
SPLIT_CONFIGS = ' POSITION                                       TOTAL-FORCE (eV/Angst)'
SPLIT_POS = ' ' + 83*'-'
ION_PATTERN = r'number of ions\s*nions\s*=\s*(\d+)'
LATTICE_PATTERN = r'direct\s*lattice\s*vectors.*\n\s*((?:\d+.\d+\s*){3}).*\n\s*((?:\d+.\d+\s*){3}).*\n\s*((?:\d+.\d+\s*){3}).*'
ENERGY_PATTERN = r'free\s*energy\s*toten\s*=\s*(-?\d+.\d+)'

'''
Hauptobjekt des packages. Es dient dazu die datei zu laden, den Inhalt zentral abzuspeichern und zu verarbeiten.
Dabei lassen sich nur files öffnen, die mit "outcar.digit" *enden*!
'''
class Parser:
    # initializiert das Objekt und lädt den Inhalt der Datei, falls es ein OUTCAR-file ist.
    def __init__(self, filepath: str):
        self.filepath = filepath
        if not search(r'outcar\.\d+', self.filepath, IGNORECASE):
            raise ValueError(f'expected outcar file, got {self.filepath}')
            
        with open(self.filepath, 'r') as outcar_in:
            self.outcar_content = outcar_in.read()
        
    # Durchsucht den Inhalt nach dem oben angegebenen Muster, sprich nach der Zeile, in der die Ionen stehen
    # Falls die entsprechende Zeile nicht auffindbar ist, bricht das Programm ab
    def find_ion_nr(self) -> int:
        ion_match = search(ION_PATTERN, 
                            self.outcar_content, 
                            IGNORECASE)
        if not ion_match:
            raise RuntimeError(f'could not find match for ions')

        return int(ion_match.group(1))


    # Durchsucht den Inhalt nach dem Muster, dass den direkten lattice Vektoren vorangestellt ist.
    # Falls die entsprechende Zeile nicht auffindbar ist, bricht das Programm ab
    def find_lattice_vectors(self) -> array:
        # TODO: ask in what shape the vectors are put in
        lattice_match = search(LATTICE_PATTERN,
                                self.outcar_content, 
                                IGNORECASE)

        if not lattice_match:
            raise RuntimeError(f'could not find match for lattice vectors')
            
        matches = lattice_match.groups()
        lattice_string = list(map(lambda vec_string: vec_string.strip().split(), matches))
        lattice_float = array(list(map(lambda list_vecs: self.__convert_list(list_vecs), lattice_string)))

        product_mat = lattice_float.dot(lattice_float.T)
        compare_mat = inner(lattice_float[:, 0], lattice_float[:, 0]) * eye(3,3)
        if not array_equal(product_mat, compare_mat):
            print(f'*************WARNING*************\nThe given lattice vectors\n{lattice_float}\n do not constitute a simple basic lattice.\n The programm wont work correctly')

        return array(lattice_float)


    # Teilt den Inhalt erst in Konfigurationen und findet die Energien, Positionen sowie Kräfte
    # und baut daraus einen Iterator
    def build_configurations(self, step_size: int) -> (float, array, array):
        # Teilt den Inhalt an den in SPLIT_CONFIGS angegebenen Zeilen, die recht zuverlässig die einzelnen
        # Konfigurationen trennen sollten, wählt anschließend jede step_size-te Konfiguration aus
        configs = self.outcar_content.split(SPLIT_CONFIGS)[1::step_size]
        for (i, config) in enumerate(configs):
            # Sucht nach der Zeile, die die Energie enthalten sollte. Falls diese nicht gefunden werden kann, abbruch
            energy_match = search(ENERGY_PATTERN, config, IGNORECASE)
            if not energy_match:
                raise RuntimeError(f'Could not find energy in config {i*step_size}')
            else:
                energy: float = float(energy_match.group(1))

            # Teilt anhand der Abtrennungen die aus "---" bestehen und wählt den Teil aus, der Pos + Kräfte enthält
            vecs_as_string: str = config.split(SPLIT_POS)[1]
            vecs_as_str_list = list(filter(lambda line: line, vecs_as_string.split('\n')))
            vecs = array(list(map(lambda line: fromstring(line.strip(), sep='\t'), vecs_as_str_list)))

            positions: array = vecs[:, 0:3]
            forces: array = vecs[:, 3:]

            if shape(positions) != shape(forces):
                raise RuntimeError(f'Shape {shape(positions)} of positions does not match shape {shape(forces)} of forces')
            else:
                yield (energy, positions, forces)

    # private Hilfsfunktion die Listen von str in Listen von floats umwandelt
    def __convert_list(self, val_list) -> list: 
        return list(map(lambda entry: float(entry.strip()), val_list))


        
if __name__ == '__main__':  
    test_in = 'OUTCAR.21'

    test_lattice = array([
                        [10.546640000 , 0.000000000,  0.000000000], 
                        [0.000000000, 10.546640000,  0.000000000], 
                        [0.000000000,  0.000000000, 10.546640000]
                        ])
            
    parser = Parser(test_in)
    nr_ions = parser.find_ion_nr()
    assert type(nr_ions) == int, f'nr of ions should be integer, is {type(nr_ions)}'
    assert nr_ions == 64, f'nr of ions should be 64, is {nr_ions}'
    assert array_equal(parser.find_lattice_vectors(), test_lattice), 'lattice vectors do not match'

    # TODO: write test for reading of pos + forces
    wrong_parser = Parser('wrong_data_outcar.21')
    wrong_parser.find_lattice_vectors()