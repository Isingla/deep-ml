import numpy as np

def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
    
    if not a or not b or not a[0] or not b[0]:
        return -1
    if len(a[0]) != len(b):
        return -1
    if any(len(row) != len(a[0]) for row in a):
        return -1
    if any(len(row) != len(b[0]) for row in b):
        return -1
    
    c = [[0] * len(b[0]) for _ in range(len(a))]

    for row in range(len(a)):
        a_row = a[row]

        for col in range(len(b[0])):
            b_col = []

            for r in range(len(b)):
                b_col.append(b[r][col])

            c[row][col] = np.dot(a_row, b_col).item()

    return c