import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
	x = np.zeros_like(b)

	r = b - A @ x
	p = r.copy()
	for _ in range(n):
		Ap = A @ p
		alpha = (r @ r) / (p @ Ap)
		x = x + alpha * p
		r_new = r - alpha * Ap
		if np.linalg.norm(r_new) < tol:
			break
		beta = (r_new @ r_new) / (r @ r)
		p = r_new + beta * p
		r = r_new
	return x


	
