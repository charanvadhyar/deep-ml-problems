import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

	a = np.array(a)
	b = np.array(b)

	m = a[0].size
	n = b.size
	if m != n:
		return -1

	return a @ b
	pass
