"Calculate Correlation Matrix"
"Write a Python function to calculate the correlation matrix for a given dataset. The function should take in a 2D numpy array X and an optional 2D numpy array Y. If Y is not provided, the function should calculate the correlation matrix of X with itself. It should return the correlation matrix as a 2D numpy array."

import numpy as np
def calculate_correlation_matrix(X, Y=None):
    X = np.array(X)
    if Y is None:
        Y = X
    else:
        Y = np.array(Y)

    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    n = X.shape[0]

    cov = (X_centered.T @ Y_centered) / (n - 1)

    std_X = X.std(axis=0, ddof=1)
    std_Y = Y.std(axis=0, ddof=1)

    denom = np.outer(std_X, std_Y)
    return cov / denom
