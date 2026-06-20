"Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode."

import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:


	a = np.array(matrix)

	if mode == 'column':
		return np.mean(a, axis = 0)
	else:
		return np.mean(a, axis= 1)


"Explanation : Numpy inbuilt method mean gives the mean if axis is not defined it gives mean of all elements of the array if axis = 0 it gives column mean and axis =1  it gives row mean"
