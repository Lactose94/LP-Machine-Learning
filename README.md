# LP-Machine-Learning
Laborpraktikum Machine Learning ML_LiqiudSI.
## outcar_parser:
The purpose of the parser is to handle loading and processing of the outcar file. It allows to read number of ions, direct lattice vectors, as well as positions, forces and energies.

---  

**Attention**:  
The program is based upon the form of given outcar file. Changes of this form will break the parser.

---
The package contains one central class:
### The parser class:
It can only be initialised with an outcar file and will throw an exception, if the file name does not end with "outcar.digit".  
The regex patterns used for parsing the file can be found in the beginning of the `outcar_parser.py` file.
#### Variables:
- **`filepath`**: Carries the path to the outcar file as string.
- **`outcar_content`**: Carries the complete content of the outcar file as string.

#### Methoden:
- **`find_ion_nr(self) -> int`**:
  Searches via regex in `outcar_content` the line 

    > ions per type = ...
  
  and extracts and returns the number of ions as integer.  
  Throws a `RuntimeError` if no line matches the regex in `ION_PATTERN`.
- **`find_lattice_vectors(self) -> np.array`**:
  Searches via regex in `outcar_content` the line 

    > direct lattice vectors ...

   and returns the following lattice vectors as numpy array.
   Throws a `RuntimeError` if no line matches the regex in `LATTICE_PATTERN`
 - **`build_configurations(self, step_size: int, offset=0) -> (float, array, array)`**:
  This **Iterator** is used to read the energy, positions of ions and forces on ions of each configuration. As input it takes the step size, i.e. how many configurations are skipped, when reading the file and the offset, i.e. at which configuration reading starts. If the offset is not choosen bigger than the maximum number of configurations, it will return at least one configuration.
  It splits `outcar_content` at the line:

    >  POSITION TOTAL-FORCE (eV/Angst)

  Iterates with the given step size and processes each configuration.  
  To do so, the line 

    > free energy TOTEN = ...

  is searched and the energy extracted. If the pattern `ENERGY_PATTERN` is not matched a `RuntimeError` is thrown.  
  After that the string corresponding to the configuration is split at the lines consisting of 83 times "-", the first one contains the position and force vectors, which are maped linewise to floats and then converted to numpy arrays, which are then split into positions and forces.  
  Should the shape of forces not match the shape of positions a `RuntimeError` is thrown.
  Finally returns the three quantities as tuple *(nergy, positions, forces)*.  
---
## Configurations
Dieses Package dient dazu die einzelnen Konfigurationen der Ionen zu speichern, und zu verarbeiten. Es beinhaltet die Configurations-Klasse, dessen Instanzen je eine Ionen-Konfiguration und ihre Eigenschaften darstellen.

---
**Achtung**:
Das Programm überprüft nicht die Plausibilität der Eingabedaten. Diese werden als richtig vorausgesetzt.

---
Das Package enthält im Wesentlichen eine Funktion und eine Klasse:
### difference(r1, r2, a=1):
Diese Funktion berechnet den Distanzvektor der Positionen r1 und r2 nach der minimal image convention. Dabei ist a eine optionale Gitterkonstante, die mitgegeben werden muss wenn in kartesischen Koordinaten (im Gegensatz zu direkten Koordinaten) gerechnet wird.

### Die Configuration Klasse:
Diese muss zumindest mit einer Positions-Matrix der Ionen initialisiert werden. Energie und Kräfte-Matrix sind optional, da diese nicht zwingend bekannt sind. Es ist auch möglich die nearesr-neigbour-tables ihrer Positionen und der Abstände gleich zu initialisieren, falls dies erwünscht ist. Die Klasse besitzt jedoch Methoden diese selbst zu berechnen. Dies gilt ebenso für die Descriptor-Koeffizienten.
#### Variablen:
- `positions` enthält die Positionsmatrix der Ionen [Ionenindex, Raumkoordinatenindex] als 2d-numpy-array(float).
- `energy` enthält die Energie der Konfiguration als float.
- `forces` enthält die Kräftematrix der Ionen [Ionenindex, Raumkoordinatenindex] als 2d-numpy-array(float).
- `differences`: Numpy array mit den Differenzvektoren zwischen alle Ionen, hat daher die shape (Nion, Nion, 3)
- `distances`: Numpy array mit den Abständen zwischen allen Ionen, hat die shape (Nion, Nion)
- `NNlist`: Hier werden die NN indices gespeichert, so dass NNlist[i] die NN-indices der nearest neighbors enthält. Ist in einer form  gespeichert, in der direkt die Werte aus dem array abgerufen werden.
- `descriptors` enthält die descriptor-Koeffizientenmatrix der Ionen [Ionenindex, qindex] als 2d-numpy-array(float).

#### Methoden:
- **init_NN(rcut, lattice)**:
  Erstellt unter Übergabe eines cutoff-Radius rcut (float in Angstrom) und des Gitters lattice (float numpy array in Angstrom) die beiden konfigurationseigenen nearest-neighbour-tables nnpositions und nndistances. Dass der cutoff-Radius sinnvoll mit der Positionsmatrix zusammenpasst, also kleiner als die halbe Gitterkonstante ist, wird dabei vorausgesetzt aber nicht überprüft!

- **get_NNdistances(i=None)**:
  Gibt die NN-Abstände des Atoms i als array aus. Wenn kein index spezifiert wird, wird eine Liste für alle Atome erstellt.
- **get_NNdifferences(i=None)**:
  Gibt die NN-Differenzvektoren des Atoms i als array aus. Wenn kein index spezifiert wird, wird eine Liste für alle Atome erstellt
- **init_descriptor(q)**:
  Erstellt unter Übergabe eines q-Vektors (float) die descriptor-Koeffizientenmatrix. Dass der cutoff-Radius sinnvoll mit dem q-Vektor zusammenpasst wird dabei vorausgesetzt und nicht überprüft. Der descriptor-Koeffizient C_i für ein Ion i berechnet sich dabei wie folgt:

    > C_i(q) = sum_j sin(q * |r_i - r_j|)

  Dabei sind r_i und r_j die Positionsvektoren der Atome i und j, und C_i ein Vektor der gleichen Länge wie der Vektor q.

#### Tests:
Mit dummy-Konfigurationen werden die einzelnen Funktionen der Klasse getestet.

---

## Kernel
This package focuses on bundeling all the kernel related functions and give the user consistent access to them.
### functions:
- **`linear_kernel(descr_list1: np.array, descr_list2: np.array) -> np.array:`** Given two arrays of descriptors, this function calculates the Kernel matrix of the linear kernel as described in equation (??) of the mathematical documentation.
- **`gaussian_kernel(descr_list1: np.array, descr_list2: np.array, sigma: float) -> np.array:`** Given two arrays of descriptors, this function calculates the Kernel matrix of the gaussian kernel as described in equation (??) of the mathematical documentation.
- **`linear_force_submat(q: np.array, config1: configuration, descriptors_array: np.array) -> np.array:`** Given the modi one configuration and one array of descriptors, this functions builds the $N_{ion} * 3$ x $N_{ion} \cdot N_{conf}$ submatrix for T in equation (??) for one fixed configuration beta in the linear case.
- **`gaussian_force_mat(q: np.array, config1: configuration, descriptors_array: np.array, sigma: float) -> np.array:`** Given the modi one configuration and one array of descriptors, this functions builds the $N_{ion} * 3$ x $N_{ion} \cdot N_{conf}$ submatrix for T in equation (??) for one fixed configuration beta in the Gaussian case.

### The Kernel class
This class is a wrapper to consistently use the choosen Kernel type for energies and forces.
#### variables:
- **`kernel`**: Holds the choosen kernel type as function.
- **`force_submat`**: Holds the function that builds part of the derivative/force matrix of the corresponding choosen kernel.
#### Methods:
- **`predict(self, qs: np.array, config: configuration, descriptors: np.array, weights: np.array, E_ave: float) -> (float, np.array)`**: Predicts the energy and forces for the given configuration. Takes as arguments the q-vector, the configuration for which one wants to predict values, the set of descriptors used in training the model, the set of weights calculated in training the model and the average energy of the configurations used to train the model. 
---
## Calibration
Dieses package bündelt die vorherigen packages und nutzt diese um das eigentliche Machine Learning durchzuführen.
Der Benutzer legt dabei die Parameter des Machine Learnings durch einträge in der Datei `user_config.json` fest. Im folgenden werden die Parameter erläutert:
- **`file_in`**: Hier wird der Pfad zum *outcar*-file, welches die Trainingsdaten enthält, eingetragen.
- **`stepsize`**: Hier gibt der Benutzer an, wie viele Konfigurationen beim Einlesen übersprungen werden sollen. Selbst wenn die Anzahl verfügbarer Konfigurationen überschritten wird, wird immer mindestens eine eingelesen.
- **`cutoff`**: Hier gibt der Benutzer den Radius der Cutoff-Sphere in Angstroem an.
- **`nr_modi`**: Gibt an, welche Länge die Descriptor-Vektoren haben sollen.
- **`lambda`**: Parameter, welcher für die Ridge-Regression genutzt werden soll.
- **`Kernel`**: Welcher Kernel für die Entwicklung der lokalen Energie genutzt werden sollen und eventuell zusätzliche Parameter, z.B. das Sigma für den gaussian Kernel  . Bisher werden nur `linear` und `gaussian` unterstützt.

