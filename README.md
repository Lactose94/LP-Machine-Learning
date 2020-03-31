# LP-Machine-Learning
Laborpraktikum Machine Learning ML_LiqiudSI.  
Das Programm besteht im wesentlichen aus den folgenden Packages:
## outcar_parser
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
  Tests können aufgerufen werden durch  
  ```bash
  python -m unittest -v tests.py
  ```
  
  ---
