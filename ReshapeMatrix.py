"Write a Python function that reshapes a given matrix into a specified shape. if it cant be reshaped return back an empty list []"

import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method

	

	a = np.array(a)

	size =a.size


	if size != new_shape[0] *new_shape[1]:
		return [];
	


	return a.reshape(new_shape)

"Expalantion: we use numpy's inbuit method to reshape the matrix but the reshape method only works if the new shape can accomodate all elements that is the product of no.of rows and columns of the new shape should be equal to the size of the matrix so we finf sixe with size mthod and check whether size is equal otherwsiwe if not we returned a empty list"
