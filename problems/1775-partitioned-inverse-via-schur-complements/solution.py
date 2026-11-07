import numpy as np

def partitioned_inverse(M, k):
    E = M[0 : k, 0: k]
    F = M[0 : k, k : ]
    G = M[k : , 0 : k]
    H = M[k :, k :]

    H_inv = np.linalg.inv(H)
    M_over_H = E - F @ H_inv @ G
    M_over_H_inv = np.linalg.inv(M_over_H)

    top_left = M_over_H_inv
    top_right = -1 * M_over_H_inv @ F @ H_inv
    bot_left = -1 * H_inv @ G @ M_over_H_inv
    bot_right = H_inv + H_inv @ G @ M_over_H_inv @ F @ H_inv

    return np.block([[top_left, top_right],
                    [bot_left, bot_right]])

def block_slogdet(M, k):
    E = M[0 : k, 0: k]
    F = M[0 : k, k : ]
    G = M[k : , 0 : k]
    H = M[k :, k :]

    H_inv = np.linalg.inv(H)
    M_over_H = E - F @ H_inv @ G
    
    sign1, log1 = np.linalg.slogdet(M_over_H)
    sign2, log2 = np.linalg.slogdet(H)

    sign = sign1 * sign2
    logabsdet = log1   +  log2

    return (sign, logabsdet)


def bordered_inverse(A_inv, b, c):
    M_over_E = c - b.T @ A_inv @ b

    if abs(M_over_E) < 0.000000000001:
        raise ValueError

    top_left = A_inv + np.outer((A_inv @ b), (b.T @ A_inv)) * 1/M_over_E
    top_right = -1 * A_inv @ b * 1/M_over_E
    bot_left = -1 * 1/M_over_E * b.T @ A_inv
    bot_right = 1/M_over_E

    top_right = top_right.reshape(-1,1)
    bot_left = bot_left.reshape(1,-1)
    bot_right = np.array([[bot_right]])

    return np.block([[top_left, top_right],
                    [bot_left, bot_right]])


    
    
