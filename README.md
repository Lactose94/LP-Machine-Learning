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
Das Package enthält im Wesentlichen eine Klasse: 
### Die Configuration Klasse:
Diese muss zumindest mit einer Positions-Matrix der Ionen initialisiert werden. Energie und Kräfte-Matrix sind optional, da diese nicht zwingend bekannt sind. Es ist auch möglich die nearesr-neigbour-tables der Indizes und der Abstände gleich zu initialisieren, falls dies erwünscht ist. Die Klasse besitzt jedoch Methoden diese selbst zu berechnen. Dies gilt ebenso für die Descriptor-Koeffizienten.
#### Variablen:
- `positions` enthält die Positionsmatrix der Ionen [Ionenindex, Raumkoordinatenindex] als numpy-array(float).
- `energy` enthält die Energie der Konfiguration als float.
- `forces` enthält die Kräftematrix der Ionen [Ionenindex, Raumkoordinatenindex] als numpy-array(float).
- `nnindices` enthält die nearest-neighbour-Indexmatrix der Ionen als list(list(integer)).
- `nndistances` enthält die nearest-neighbour-Abständematrix der Ionen als list(list(float)).
- `descriptors` enthält die descriptor-Koeffizientenmatrix der Ionen [Ionenindex, qindex] als numpy-array(float).

#### Methoden:
- **init_NN(rcut)**:  
  Erstellt unter Übergabe eines cutoff-Radius rcut (float in Angstrom) die beiden konfigurationseigenen nearest-neighbour-tables nnindices und nndistances. Dass der cutoff-Radius sinnvoll mit der Positionsmatrix zusammenpasst wird dabei vorausgesetzt und nicht überprüft.
- **init_descriptor(q)**:  
  Erstellt unter Übergabe eines q-Vektors (float) die descriptor-Koeffizientenmatrix. Dass der cutoff-Radius sinnvoll mit dem q-Vektor zusammenpasst wird dabei vorausgesetzt und nicht überprüft. Der descriptor-Koeffizient C_i für ein Ion i berechnet sich dabei wie folgt:
  
    > C_i(q) = sum_j sin(q * |r_i - r_j|)
  
  Dabei sind r_i und r_j die Positionsvektoren der Atome i und j, und C_i ein Vektor der gleichen Länge wie der Vektor q.
  
### Tests:
Tests für dieses Package sind vorgesehen, wurden jedoch noch nicht spezifiziert.

---
## Calibration
Dieses package bündelt die vorherigen packages und nutzt diese um das eigentliche Machine Learning durchzuführen.  
Der Benutzer legt dabei die Parameter des Machine Learnings durch einträge in der Datei `user_config.json` fest. Im folgenden werden die Parameter erläutert:
- `file_in`: Hier wird der Pfad zum *outcar*-file, welches die Trainingsdaten enthält, eingetragen.
- `file_out`: Hier kann der Benutzer angeben, in welcher Datei die Ergebnisse des Learnings eingetragen werden sollen. Wie genau diese aussehen, muss noch spezifiert werden.
- `stepsize`: Hier gibt der Benutzer an, wie viele Konfigurationen beim Einlesen übersprungen werden sollen. Selbst wenn die Anzahl verfügbarer Konfigurationen überschritten wird, wird immer mindestens eine eingelesen.
- `cutoff`: Hier gibt der Benutzer den Radius der Cutoff-Sphere an.
- `nr_modi`: Gibt an, welche Länge die Descriptor-Vektoren haben sollen.
- `lambda`: Parameter, welcher für die Ridge-Regression genutzt werden soll.
- `Kernel`: Welcher Kernel für die Entwicklung der lokalen Energie genutzt werden sollen und eventuell zusätzliche Parameter, z.B. das Sigma für den gaussian Kernel  . Bisher werden nur `linear` und `gaussian` unterstützt.

---
## Kernel
Dieses Package versucht die wesentlichen Funktionalitäten des Kernels zu fokussieren.
### Funktionen:
- ¸`linear_kernel(descriptor1: np.array, descriptor2: np.array) -> float`
- `gaussian_kernel(descriptor1: np.array, descriptor2: np.array, sigma: float) -> float`
### Die Kernel-Klasse
Diese dient als wrapper um die übrigen Funktionalität dem User gebündelt zur Verfügung zu stellen
## Variablen:
- `kernel`: Enthält den gewünschten Kernel als Funktion.
- `matrix_element`: supply a set of descriptors for a configuration and **one element** of the set of **all** available configuration (vectors) and retrieve the corresponding matrix-element wrt the given kernel.