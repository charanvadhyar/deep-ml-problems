import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here


    ATA = A.T @ A


    eigvals_v,V = np.linalg.eigh(ATA)

    idx =np.argsort(eigvals_v)[::-1]

    eigvals_v = eigvals_v[idx]
    V = V[:,idx]

    singularvalues = np.sqrt(eigvals_v)

    m,n = A.shape
    sigma = np.zeros((m,n))

    for i in range(min(m,n)):
        sigma[i,i] = singularvalues[i]

    U = (A @ V) / singularvalues
    VT = V.T 
    svd = U @ sigma @ VT


    return U,singularvalues,VT

    pass
