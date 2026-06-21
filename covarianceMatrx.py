import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	
	a = np.array(vectors)
	n = a.shape[1]

	a_centered = a - a.mean(axis = 1, keepdims=True)

	return (a_centered @ a_centered.T)/(n-1)
