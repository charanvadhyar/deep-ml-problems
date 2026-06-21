import numpy as np


def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	a = np.array(dense_matrix)

	row_idx, col_idx = np.nonzero(a)
	vals = a[row_idx,col_idx]
	counts = np.bincount(row_idx, minlength=a.shape[0])   
	row_ptr = np.concatenate(([0], np.cumsum(counts)))

	return vals.tolist(), col_idx.tolist(), row_ptr.tolist()


"np.nonzero(A) scans row-major and returns the row and column of every nonzero, in that scan order — that ordering is exactly why values comes out correctly sequenced without any extra sorting.
A[row_idx, col_idx] is paired fancy indexing — it doesn't give you a grid, it gives you A[row_idx[0],col_idx[0]], A[row_idx[1],col_idx[1]], ... — one value per matched pair, which is exactly "pull out the nonzero values in order."
np.bincount(row_idx, ...) counts how many times each row index appears — i.e. how many nonzeros each row has. minlength=A.shape[0] matters: without it, a matrix whose last row happens to be all zeros would silently produce a too-short array, since bincount only goes as high as the max value seen.
np.cumsum turns "count per row" into "running total so far" — and prepending 0 gives you the starting offset for row 0, which is always 0."
"Sparse embeddings/retrieval (TF-IDF, BM25, sparse vector search) — exactly the sparse-matrix problems on your own Deep-ML list (#60, #65, #67). Term-document matrices are almost entirely zeros; CSR is the standard way to store them.
Graph/adjacency matrices — most real graphs are sparse; CSR (or its column-oriented cousin CSC) is the default storage in graph libraries.
scipy.sparse.csr_matrix — the real-world tool; this is the production implementation you'd actually call. Building it once by hand demystifies what that object is doing internally and why sparse matmul is fast (it skips every zero entirely, rather than multiplying by zero and wasting the cycle).
Large-scale linear algebra generally — this is the data structure that makes the conjugate gradient method you just built actually practical at scale: huge sparse A matrices in CG/FEM/physics solvers are stored exactly this way."
