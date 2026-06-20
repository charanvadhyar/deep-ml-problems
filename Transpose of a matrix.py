"Write a Python function that computes the transpose of a given 2D matrix. The transpose of a matrix is formed by turning its rows into columns and columns into rows. For an mÃn matrix, the transpose will be an nÃm matrix."

import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    a = np.array(a)

    return a.T
    
    pass

"we used numpy built in method to Transpsoe a matrix which is array.T"
