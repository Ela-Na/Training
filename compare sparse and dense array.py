# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:31:37 2020

@author: Ela
"""

import numpy as np
from scipy import sparse

n_rows = 10000
n_cols = 10000

example = np.random.binomial(1, p=0.05, size=(n_rows, n_cols))

print(f"size of dense array: {example.nbytes}")

sparse_example = sparse.csr_matrix(example)

print(f"size of sparse array: {sparse_example.data.nbytes}")


full_size = (sparse_example.data.nbytes + 
      sparse_example.indptr.nbytes +
      sparse_example.indices.nbytes)

print(f"full size of sparse array: {full_size}")
