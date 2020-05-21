import numpy as np
"""
import configuration as conf
from time import time
from sklearn.linear_model import Ridge

##### Provided input data #####
rcut = 4
q = [np.pi/rcut,2*np.pi/rcut,3*np.pi/rcut]
Ni = 4
Na = 2
Nq = 3
lattice = np.array([[10.0 ,  0.0],
                    [ 0.0 , 10.0]])
positions1 = np.array([[1.0 , 1.0],
                       [6.0 , 9.0],
                       [8.0 , 8.0],
                       [9.0 , 6.0]])
positions2 = np.array([[1.0 , 1.0],
                       [3.0 , 3.0],
                       [1.0 , 9.0],
                       [9.0 , 9.0]])
energy1 = 1.2
energy2 = 1.4
forces1 = np.array([[ 0.1 ,  0.1],
                    [-0.3 ,  0.1],
                    [ 0.1 ,  0.1],
                    [ 0.1 , -0.3]])
forces2 = np.array([[ 0.0 ,  0.2],
                    [ 0.2 ,  0.2],
                    [ 0.0 , -0.2],
                    [-0.2 , -0.2]])

config1 = conf.Configuration(positions1, energy1, forces1)
config2 = conf.Configuration(positions2, energy2, forces2)

config1.init_nn(rcut, lattice)
config2.init_nn(rcut, lattice)

config1.init_descriptor(q)
config2.init_descriptor(q)
#print(config1.descriptors)
#print(config2.descriptors)

##### Put configuration data in supermatrices #####
C = np.empty([Ni*Na, Nq])
C[0:Ni,:] = config1.descriptors
C[Ni:2*Ni,:] = config2.descriptors
E = np.array([[energy1],[energy2]])
#print(C)

##### linear Kernel #####
K1 = np.matmul(C,np.transpose(C)) # K[alpha', alpha, i', i] = K[alpha'*4+i', alpha*4+i]
#print(K1)

##### Preparing system of linear equations for energy #####
K2 = np.empty([Na, Ni*Na]) # K2 is K1 summed over i'
for l in range(Ni*Na):
    for k in range(Na):
        K2[k,l] = np.sum(K1[k*Ni:(k+1)*Ni,l])
#print(K2)
X = np.matmul(np.transpose(K2),K2)
#print(X)
y = np.matmul(np.transpose(K2),E)
lamb = 0.5
# y = X*w + lamb*w     ->     w = ?
# w = (X + lamb*I)^(-1) * y

##### Ridge Regression for energy

### mit ridge-Paket
start = time()
clf = Ridge(alpha=lamb) # muss sklearn installieren
clf.fit(X, y)
w = clf.coef_
end = time()
print("time passed for ridge packet:", end-start)
print(w)

### mit 0815 numpy funktionen
#start = time()
w = np.matmul(np.linalg.inv(X + lamb * np.eye(Ni*Na)) , y)
#end = time()
#print("time passed for conservative inversion:", end-start)
print(w)

##### Anwendung auf neue Konfiguration #####
#positions3 = np.array([[2.0 , 2.0],
#                       [2.0 , 8.0],
#                       [8.0 , 2.0],
#                       [8.0 , 8.0]])
config3 = conf.Configuration(positions1)
config3.init_nn(rcut, lattice)
config3.init_descriptor(q)
C_3 = config3.descriptors
#print(C_3)
K1_3 = np.matmul(C_3,np.transpose(C))
#print(K1_3)
K2_3 = np.sum(K1_3, axis=0)
print(K2_3)

### mit 0815 numpy funktionen

#coeffA = 1/np.tensordot(np.transpose(K2_3), K2_3, axes=1) * lamb + 1
#E_3 = coeffA * np.matmul(w,K2_3.reshape(-1, 1))
E_3 = np.matmul(w,K2_3.reshape(-1, 1))
print(E_3)

### mit ridge-Paket
E_3 = clf.predict(K2_3.reshape(1, -1))
print(E_3)
"""

##### Gaussian Kernel mit nur numpy funktionen #####

#ar1 = np.array([[1.0,2.0],[3.0,4.0],[5.0,6.0]])

ar1 = np.array([[1.0,2.0,3.0,4.0,5.0],
                [3.0,4.0,5.0,6.0,7.0],
                [5.0,6.0,7.0,8.0,9.0],
                [7.0,8.0,9.0,10.,11.],
                [9.0,10.,11.,12.,13.],
                [11.,12.,13.,14.,15.],
                [13.,14.,15.,16.,17.],
                [15.,16.,17.,18.,19.]])

#ar2 = ar1.copy()

ar2 = np.array([[1.0,2.0,3.0,4.0,5.0],
                [3.0,4.0,5.0,6.0,7.0],
                [5.0,6.0,7.0,8.0,9.0],
                [7.0,8.0,9.0,10.,11.]])

(ni,nq) = np.shape(ar1)
nj = np.shape(ar2)[0]
ar3 = np.empty((ni,nj))
for i in range(ni):
    for j in range(nj):
        ar3[i,j] = np.sum((ar2[j,:]-ar1[i,:])**2)
#print(ar3) # so sollte die gaussian-kernelmatrix aussehen
ar4 = np.sum(np.transpose(np.diagonal( (np.subtract.outer(ar2, ar1))**2 , axis1=1, axis2=3), axes=(1,0,2)), axis=2) # gaussian-kernelmatrix mit numpy-funktionen
#print(ar4) 
print(np.allclose(ar3, ar4))

ar5 = np.empty((ni,nj,nq))
for i in range(ni):
    for j in range(nj):
        for q in range(nq):
            ar5[i,j,q] = ar2[j,q] - ar1[i,q]
#for q in range(nq):
#    print(ar5[:,:,q]) # so sollte die descriptor-differentmatrix aussehen
ar6 = np.sum(ar5**2, axis=2)
#print(ar6) # das ist equivalent zu ar4 -> yes!
print(np.allclose(ar4, ar6))

ar7 = np.transpose(np.diagonal(np.subtract.outer(ar2, ar1), axis1=3, axis2=1), axes=(1,0,2)) # descriptor-differentmatrix mit numpy funktionen
#for q in range(nq):
#    print(ar7[:,:,q]) 
ar8 = np.sum(ar7**2, axis=2)
#print(ar8) # das ist equivalent zu ar4 -> yes!

print(np.allclose(ar5, ar7))
print(np.allclose(ar4, ar8))


##### Gaussian Kernel mit 1 Loop für q #####

ar9 = np.empty((ni,nj,nq)) # descriptor-differentmatrix mit 1 Loop für q (eventuell schneller, loop vs. unnötige elemente ?)
for q in range(nq):
    ar9[:,:,q] = np.transpose(np.subtract.outer(ar2[:,q], ar1[:,q]))
#    print(ar9[:,:,q])
ar10 = np.sum(ar9**2, axis=2) # gaussian-kernelmatrix mit 1 Loop für q
#print(ar10) # das ist equivalent zu ar4 -> yes!
print(np.allclose(ar7, ar9))
print(np.allclose(ar4, ar10))
