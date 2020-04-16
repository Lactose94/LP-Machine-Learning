# LP-Machine-Learning
Laborpraktikum Machine Learning ML_LiqiudSI.  
Das Programm besteht im wesentlichen aus den folgenden Packages:
## Parser
Der Parser dient dazu das outcar-file zu laden und so zu verarbeiten, dass Ionenzahl, direkte Gittervektoren, sowie Positionen und Kräfte ausgelesen und vom Hauptprogramm weiterverarbeitet werden können.

---
**Achtung**: 
Das Programm basiert sehr stark auf der Form des Beispielfiles. Änderungen an dieser Form können das komplette Package brechen.

---
Zu Beginn des Packages werden als Konstanten die Patterns für Regex und das Splitting angelegt. Änderungen an der Formatierung des Files werden sich also hier wiederspiegeln.  
Das Package enthält im Wesentlichen eine Klasse: 
### Die Parser Klasse:
Diese kann ausschließlich mit einem outcar-file initialisiert werden. Tatsächlich wird eine exception geworfen, falls der Dateiname nicht auf "outcar.digit" endet. 
#### Variablen:
- `filepath` enthält den Pfad zum outcar-file als String.
- `outcar_content` enthält den kompletten Inhalt des Files als String.

#### Methoden:
- **find_ion_nr**:  
  Durchsucht mithilfe von Regex den Inhalt nach der Zeile  
  
    > ions per type = ...  
    
  extrahiert daraus die Ionenzahl und gibt diese als Integer zurück.   
  Wirft einen `RuntimeError` falls keine solche Zeile gefunden werden kann.
- **find_lattice_vectors**:  
  Durchsucht mithilfe von Regex den Inhalt nach der Zeile:
    
    > direct lattice vectors ...
   
   und gibt die darauf folgenden lattice vectors als numpy array zurück.  
   Wirft einen `RuntimeError`falls keine solche Zeile gefunden werden kann.
 - **build_configurations**:  
  Dieser **Iterator** dient dazu die Werte der einzelnen Konfigurationen auszulesen.  
  Nimmt als input die Schrittweite, wie viele Konfigurationen übersprungen werden sollen, gibt jedoch immer mindestens eine Konfiguration zurück.  
  Spaltet zuerst den Inhalt an der Zeile:
    
    >  POSITION TOTAL-FORCE (eV/Angst)  
   
   Iteriert über die entsprechende Schrittweite und verarbeitet darauf jede Konfiguration einzeln.  
  Dazu wird zuerst nach der Zeile  
      
    > free energy TOTEN = ... 
    
    gesucht und die Energie extrahiert. Falls die Zeile nicht gefunden werden kann, wird ein `RuntimeError` geworfen.    
  Anschließend wird der Text der Konfiguratition an den Linien bestend aus einem Leerzeichen und 83 mal "-" aufgespalten. Die erste davon enthält die Positions- und Kraftvektoren, welche Zeilenweise in floats und dann in numpy arrays umgewandelt und anschließend als Positionen und Kräfte getrennt in arrays gespeichert werden.  
  Sollte dabei die shape der Kräfte nicht mit der der Positionen übereinstimmen, wird eine `RuntimeError` geworfen.  
  Schließlich werden diese drei Werte als Tupel zurück gegeben in der Form *(E, Positionen, Kräfte)*.  
  
  ### Tests:
  Führt man das File einzeln aus, werden nacheinander kleinere Assertions überprüft und anschließend über alle Konfigurationen des Beispielfiles iteriert und auf dem Bildschirm ausgegeben. Sollte in Zukunft noch durch ein vernünftiges Testframework ersetzt werden.
  
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
  
### Tests:
Mit dummy-Konfigurationen werden die einzelnen Funktionen der Klasse getestet.

---
## Calibration
Dieses package bündelt die vorherigen packages und nutzt diese um das eigentliche Machine Learning durchzuführen.  
Der Benutzer legt dabei die Parameter des Machine Learnings durch einträge in der Datei `user_config.json` fest. Im folgenden werden die Parameter erläutert:
- `file_in`: Hier wird der Pfad zum *outcar*-file, welches die Trainingsdaten enthält, eingetragen.
- `file_out`: Hier kann der Benutzer angeben, in welcher Datei die Ergebnisse des Learnings eingetragen werden sollen. Wie genau diese aussehen, muss noch spezifiert werden.
- `stepsize`: Hier gibt der Benutzer an, wie viele Konfigurationen beim Einlesen übersprungen werden sollen. Selbst wenn die Anzahl verfügbarer Konfigurationen überschritten wird, wird immer mindestens eine eingelesen.
- `cutoff`: Hier gibt der Benutzer den Radius der Cutoff-Sphere in Angstroem an.
- `nr_modi`: Gibt an, welche Länge die Descriptor-Vektoren haben sollen.
- `lambda`: Parameter, welcher für die Ridge-Regression genutzt werden soll.
- `Kernel`: Welcher Kernel für die Entwicklung der lokalen Energie genutzt werden sollen und eventuell zusätzliche Parameter, z.B. das Sigma für den gaussian Kernel  . Bisher werden nur `linear` und `gaussian` unterstützt.

---
## Kernel
Dieses Package versucht die wesentlichen Funktionalitäten des Kernels zu fokussieren.
### Funktionen:
- `linear_kernel(descr_list1: np.array, descr_list2: np.array) -> np.array:` Given two arrays of descriptors, this function calculates the Kernel matrix wrt the linear kernel.
- `gaussian_kernel(descr_list1: np.array, descr_list2: np.array, sigma: float) -> np.float:` Given two arrays of descriptors, this function calculates the Kernel matrix wrt the linear gaussian kernel.
- `grad_scalar(q: float, dr: np.array) -> np.array:` The gradient of a descriptor is the sum over a scalar prefactor times a difference-vector. This function builds the scalar prefactors from an array of distances.
- `def linear_force_subrow(q: np.array, config1: configuration, descriptors_array: np.array) -> np.array:` Builds the force matrix elements for config1 given a set of descriptors, in the *linear* case. For example, if we have two configurations, then the function will build the matrix element T^(1,k=1,...,Nions, l=1...,3)_(2,i=1,...,Nions).
If given a complete set of descriptors i.e (Nconf * Nions, n_q) it builds the rows of the coefficient matrix for the given configuration.

### Die Kernel-Klasse
Diese dient als wrapper um die übrigen Funktionalität dem User gebündelt zur Verfügung zu stellen
#### Variablen:
- `kernel`: Enthält den gewünschten Kernel als Funktion.
#### Methods
- `energy_matrix_element`: supply two sets of descriptors, from which the corresponding matrixelements are built. For example, if descriptors of config1 and config2 are given, the output will be sum_i K^(1,i=1,..,Nions)_(2,j=1,...,Nions). If we supply one configuration and a full set of descriptors, then we get the corresponding row, which multiplied by the weights should give the energy.
- `force_submat`: takes as input two configurations and just applies the correct submatrix (linear or gaussian) to them.