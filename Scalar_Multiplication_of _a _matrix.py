"Write a Python function that multiplies a matrix by a scalar and returns the result."

import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:

	a =np.array(matrix)

	return a * scalar


"Explanation : Numpy gives the option of mutiplying a scalar directly to matrix"
