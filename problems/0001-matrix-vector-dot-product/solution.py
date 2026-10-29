import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	res = []
	if len(a) != len(b):
		return -1
	for vec in a:
		res.append(np.dot(vec, b))
	return res