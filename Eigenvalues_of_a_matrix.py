"Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest."

import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:


	a = np.array(matrix)

	return np.linalg.eigvals(a)


"Explanation : the inbuilt linalg library is used for linear algebra operations and linaglg.eig will return both eignevalues and eigenvectors if we need only eigenvalues we can do linalg.eigvals but remember there is no method to find only eigenvectos if we need only eigen vectoes we can run linalg.eig and access them remeber eigen vectors are stored as columns not roews if we need specidfic eienevcotr we can use slicng like for first eigenvectoer we can do rigenvectors[:,0]"
