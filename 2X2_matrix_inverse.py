"Write a Python function that calculates the inverse of a 2x2 matrix. The inverse of a matrix A is another matrix A_inv such that A * A_inv = I (the identity matrix).\n\nFor a 2x2 matrix [[a, b], [c, d]], the inverse exists only if the determinant (ad - bc) is non-zero.\n\nReturn None if the matrix is not invertible (i.e., when the determinant equals zero)."

import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """

    a = np.array(matrix)

    det = np.linalg.det(a)

    if det == 0 :
        return None

    return np.linalg.inv(a)
