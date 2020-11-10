# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:26:43 2020

@author: Ela
"""

import numpy as np
from scipy import sparse

example = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [1, 0, 1]])

sparse_example = sparse.csr_matrix(example)

print(sparse_example.data.nbytes)

print(sparse_example.data.nbytes + 
      sparse_example.indptr.nbytes +
      sparse_example.indices.nbytes)