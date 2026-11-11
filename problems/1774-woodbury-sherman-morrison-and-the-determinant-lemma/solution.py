import numpy as np

def woodbury_solve(sigma_diag, X, B):

    sigma_inv = 1/sigma_diag
    SinvX = sigma_inv.reshape(-1, 1) * X

    I = np.eye(X.shape[1])
    inner = I + X.T @ SinvX 

    was_flat = B.ndim == 1
    if B.ndim == 2:
        SinvB = sigma_inv.reshape(-1,1) * B
    else:
        SinvB = sigma_inv.reshape(-1,1) * B.reshape(-1,1)

    XtSinvB = X.T @ SinvB
    mid = np.linalg.solve(inner, XtSinvB)

    ans = SinvB - SinvX @ mid
    if was_flat:
        return ans.reshape(-1)
    return ans


def sherman_morrison(A_inv, u, v):
    col = A_inv @ u
    row = v.T @ A_inv
    denom = 1 + v.T @ col

    if denom == 0:
        raise ValueError
    
    correction = np.outer(col, row)

    return A_inv - correction / denom

def logdet_rank_one_update(logdet_A, A_inv, u, v):
    denom = 1 + v.T @ A_inv @ u

    if denom <= 0:
        raise ValueError
    
    return float(logdet_A + np.log(denom))

