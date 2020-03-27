from re import match, search, IGNORECASE
from numpy import array



# Object which abstracts the parsing utilizing regex
class Parser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        if not search(r'outcar\.\d+', self.filepath, IGNORECASE):
            raise ValueError(f'expected outcar file, got {self.filepath}')
            
        with open(self.filepath, 'r') as outcar_in:
            self.outcar_content = outcar_in.read()
        

    def find_ion_nr(self) -> int:
        ion_match = search(r'number of ions\s*nions\s*=\s*(\d+)', 
                            self.outcar_content, 
                            IGNORECASE)
        if not ion_match:
            raise RuntimeError(f'could not find match for ions')

        return int(ion_match.group(1))


    def find_lattice_vectors(self) -> array:
        # TODO: ask in what shape the vectors are put in
        lattice_match = search(r'direct\s*lattice\s*vectors.*\n\s*((?:\d+.\d+\s*){3}).*\n\s*((?:\d+.\d+\s*){3}).*\n\s*((?:\d+.\d+\s*){3}).*',
                                self.outcar_content, 
                                IGNORECASE)

        if not lattice_match:
            raise RuntimeError(f'could not find match for lattice vectors')
            
        matches = lattice_match.groups()
        lattice_string = map(lambda vec_string: vec_string.strip().split(" "), matches)
        lattice_float = map(lambda list_vecs: self.__convert_list(list_vecs), lattice_string)

        return array(lattice_float)


    # TODO: write function that splits content into N^alpha tries and retrives the n-th positions and forces and energies
    # Helper function to convert list to float
    def __convert_list(self, val_list) -> list: 
        return list(map(lambda entry: float(entry.strip()), val_list))
            

        
    