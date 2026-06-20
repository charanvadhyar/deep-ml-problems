"Write a Python function that transforms a given matrix A using the operation 
 T^−1AS, where T and S are invertible matrices. The function should first validate if the matrices T and S are invertible, and then perform the transformation. In cases where there is no solution return -1"

import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

	det_t = np.linalg.det(T)
	det_s = np.linalg.det(S)

	if det_t == 0 or det_s == 0:
		return -1

	
	inverse_t = np.linalg.inv(T)


	intermediate_matrix = np.matmul(inverse_t, A)

	transformed_matrix = np.matmul(intermediate_matrix, S)	



	return transformed_matrix

"Explaantion : we use linalg.det to find the deternminat of both the matrices if the det is zero they are singular matrices so they inverse wom't exist then we sue linalg.inv method to find inverse and use matmul to do transformatons"
