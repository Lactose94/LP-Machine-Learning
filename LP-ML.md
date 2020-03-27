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
Ab hier spaltet sich die Definition, jenachdem ob die (später definierten) Variante 1 oder 2 gewählt würde.
##### Variante 1:
```python
  def process_content(self, config_nr: int) -> np.array[float; Nalpha x Nions]:
```

Nimmt als input die Anzahl der einzulesenden Konfigurationen und verarbeite den Inhalt des files so, dass ein riesiges numpy array nach der Form $[[E_1, \vec{r_1}^N, \vec{F_1}^N], [E_2, \vec{r_2}^N, \vec{F_2}^N],...]$ mit den zugehörigen Energien, Postionen und Kräften zurückgegeben würde. Für die späteren Objekte würde dann nur die jeweilige Zeile etc. gespeichert werden, in der die zugehörige Größe zu finden ist.
##### Variante 2
```Python
def process_content(self, config_nr: int) -> array[configuration; Nalpha]:
```
Verarbeitet die Daten wie vorher, nur dass die Daten sofort als config-objekt gespeichert würden und diese als Array zurückgegeben werden.

----------
## Variante 1:
**Problem:** Wie bekommen die untergeordneten objekte an die Daten vom zentralen Array?
Werte werden nur zentral in einem Array gespeichert, alle folgenden Objekte tragen nur die Referenz, bzw. den Index, wo das spezifische Datum im Array zu finden ist.

---
### Ion:
```Python
class Ion:
  # enthält index/indice für Ort und Kraft Vektor im zentralen Vektor.
  self.index: int
  # enthält ein array mit den indices von den nächsten Nachbarn.
  self.nearest_neigbhbors: np.array[int; nr_ions]
  # enthält die zu berechnenden descriptoren für das jeweilige Ion. Eventuell
  # auch hier eher eine Referenz zu einem zentralen Vektor.
  self.descriptors: np.aray[float; max_n]
```
#### Methoden:
```Python
def check_nn(self, ion: Ion, r_cutoff: float) -> None:
```
Ruft von beiden Ionen die Position ab und überprüft ob $r_{ij}<R_{cut}$. Falls ja, wird bei beiden Ionen der Index des jeweiligen anderen eingesetzt.

---
```python
def calc_descriptors(self, max_n: int) -> None:
```
Ruft die Liste der NN ab und darauf deren Positionen, aus denen die Deskriptoren gebildet werden.

---
### Configuration:
```Python
class Configuration:
  self.cutoff: foat
  self.ion_list: array[Ion; Nion]
  # enthält den Index der configuration im Supervektor.
  self.index: int
  # maximales n bis zu dem descriptoren ausgwertet werden
  self.max_n: int
  ```
  #### Methoden:
  ```python
  def calc_nn(self) -> None:
  ```
  geht die Liste der Ionen durch und ruft nacheinander die entsprechende Methode um zu überprüfen, ob es sich um NN handelt.

---
```python
def calc_descriptors(self) -> None:
```
Geht Liste der Ionen durch und berechnet jeweils die descriptoren durch die entsprechende function der Ionen.

---
## Variante 2:
Werte werden in kleineren Einheiten gespeichert, entweder pro Ion selbst b) oder ein mal als Vektor in jeder configuration a).

### Variante 2a):
---
### Ion:
```Python
class Ion:
  # enthält index/indice für Ort und Kraft Vektor im zentralen Vektor.
  self.index: int
  # enthält ein array mit den indices von den nächsten Nachbarn.
  self.nearest_neigbhbors: np.array[int; nr_nn]
  # enthält die zu berechnenden descriptoren für das jeweilige Ion. Eventuell
  # auch hier eher eine Referenz zu einem zentralen Vektor.
  self.descriptors: np.aray[float; max_n]
```
#### Methoden:
Erwähnt werden hier nicht diejenigen Funktionen, die zum setzen und nehmen der jeweiligen Werte benötigt werden.

---
```python
def add_nn(self, ion: Ion) -> None:
```
falls ein Ion als NN identifiziert wurde, fügen den Index den nächsten Nachbarn hinzu.

---
``` python
def calc_descriptors(self, own_position: np.array[float; 3], nn_positions: np.array[float; 3xnr_NN], max_n: int) -> None:
```
Nimmt die Position des Ions und der anderen Ionen und das maximale n für die qs als Input und berechnte daraus die descriptoren und setzt sie für das Ion.

---
### Configuration:
```Python
class Configuration:
  self.energy: float
  self.positions: np.array[float; 3xn_ions]
  self.forces: np.array[float; 3xn_ions]
  self.ion_list: array[Ion: Nions]
  self.cutoff: float
  self.max_n: int

```
#### Methoden:
---
```python:
def calc_nn(self, cutoff: float) -> None:
```
Geht die Liste der Ionen durch, holt ihre Positionen und schaut, ob zwei Ionen NN sind (bez. $R_{cut}$) und setzt entsprechend die Werte.

---

```python
def calc_descriptors(self, max_n) -> None:
```
Geht die Ionen durch und holt pro Ion die Liste der NN und dementsprechend die Positionen und ruft dann die entsprechende funktion des Ions auf.

---
### Variante 2b):
Alle Werte werden lokal im Ion gespeichert, die Configuration dient nur als organisator.  
Hat den Vorteil, dass kein Problem mit Ordnung entstehen kann.
### Ion:
```python
class Ion:
  self.index
  self.position
  self.force
  self.nn_list # enthält Liste mit ids/indices der NN
  self.decriptors
```

---
#### Methoden:
---
```python
def check_nn(self, ion: Ions, cutoff: float) -> None:
```
Siehe oben

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
  self.ion_list: array[Ion: Nions]
  self.cutoff: float
  self.max_n: int
```
#### Methoden:
---
```python:
def calc_nn(self) -> None:
```
Geht Liste der Ionen durch und ruft paarweise die entsprechende function der Ionen auf.

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