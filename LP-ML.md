# Design ML-LP
------
### Parser
Package zum lesen und verarbeiten des outcar-files, implementiert mithilfe von 'regex'.
soll folgendermaßen organisiert sein:
```Python
class Parser:
  # Variables
  self.filepath: str
  self.content: str
```
Beim initialisieren wird direkt das file gelesen und der Inhalt (roh) in content abgelegt.
#### Methoden:
```python
  def find_ion_nr(self) -> int:
```
Sucht mithilfe von regex nach den Keywords hinter denen die Ionen-Anzahl steht und gibt sie als integer zurück.

---
```python
  def find_lattice_vec(self) -> np.array:
```
Sucht mithilfe von regex nach den Keywords unter denen die lattice-vectors zu finden sind und gibt sie als numpy array zurück.

---

```Python
def process_content(self, config_nr: int) -> (float, array, array):
```
Als Iterator implementiertes Verarbeiten der configurations. Gibt jeweils die Werte von einer Konfiguration in der Form (E, R, F) aus.

--- 

Alle Werte werden lokal im Ion gespeichert, die Configuration dient nur als Organisator.  
Hat den Vorteil, dass kein Problem mit Ordnung entstehen kann.
### Ion:
```python
class Ion:
  self.index: int
  self.position: array
  self.force: array
  self.nn_list: list # enthält Liste mit ids/indices der NN
  self.decriptors: array
```

---
#### Methoden:
---
```python
def check_nn(self, ion: Ions, cutoff: float) -> None:
```
prüft ob der Abstand zum Ianderen on kleiner als der cutoff ist. Falls ja, wird für beide der Index des jeweils anderen als nächster Nachbar eingetragen.

---
```python
def calc_descriptors(self, nn: array[Ion: nr_nn], max_n: int) -> None:
```
Geht die Liste der NN durch und berechnet entsprechend die Deskriptoren

---
### Configuration:
```Python
class Configuration:
  self.energy: float
  self.ion_list: list[Ion: Nions]
  self.cutoff: float
  self.max_n: int
```
#### Methoden:
---
```python:
def calc_nn(self) -> None:
```
Geht Liste der Ionen durch und ruft paarweise die entsprechende Funktion der Ionen auf.

---
```python
def calc_descriptors(self, max_n: int) -> None:
```
Geht die Liste der Ionen durch, holt von jedem die Liste der NN und übergibt die entsprechenden Ionen der calc_descriptors function des Ions.

---
```python
def get_all(self) -> (float, np.array, np.array, np.array):
```
vektorisiert die lokalen Daten und gibt sie aus als Tupel $(E_1, \vec{r_1}^N, \vec{F_1}^N, \vec{C}^N)$
