import numpy as np
from matmul import matmul

A = np.array([
    [1,2],
    [3,4]
])

B = np.array([
    [5,6],
    [7,8]
])

print(matmul(A,B))