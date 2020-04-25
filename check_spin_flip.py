import numpy as np
import random

pauli_x = np.array([ 
    [0,1],
    [1,0]
])
pauli_y = np.array([ 
    [0,complex(0,-1)],
    [complex(0,1),0]
])
pauli_z = np.array([ 
    [1,0],
    [0,-1]
])

sigma_sigma_3d = np.add( np.add(np.kron(pauli_x, pauli_x) , np.kron(pauli_y, pauli_y) ), np.kron(pauli_z, pauli_z) )
print(sigma_sigma_3d.tolist())

