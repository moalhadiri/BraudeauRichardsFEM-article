"""Consistent mass matrix on the native tetrahedral mesh."""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..basisfunctions.tetrahedra import kpde3dgphi

def kpde3dmass(p, t, alfa):
    """Assemble the consistent P1 mass matrix with a scalar, nodal or cell coefficient."""

    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=int)

    np_nodes = p.shape[0]
    nt = t.shape[0]

    alfa = np.asarray(alfa, dtype=float)

    if alfa.size == 1:
        c = np.full(nt, float(alfa.reshape(-1)[0]))

    elif alfa.size == nt:
        c = alfa.reshape(-1).copy()

    elif alfa.size == np_nodes:
        alfa = alfa.reshape(-1)

        c = (alfa[t[:, 0]] + alfa[t[:, 1]] + alfa[t[:, 2]] + alfa[t[:, 3]]) / 4.0

    else:
        raise ValueError(
            "alfa doit être un scalaire, un vecteur de taille nt "
            ""
        )

    vol, *_ = kpde3dgphi(p, t)

    c = c * vol / 20.0

    rows = []
    cols = []
    data = []

    for i in range(4):
        for j in range(i):
            rows.extend(t[:, i])
            cols.extend(t[:, j])
            data.extend(c)

            rows.extend(t[:, j])
            cols.extend(t[:, i])
            data.extend(c)

    for i in range(4):
        rows.extend(t[:, i])
        cols.extend(t[:, i])
        data.extend(2.0 * c)

    M = coo_matrix(
        (data, (rows, cols)),
        shape=(np_nodes, np_nodes)
    ).tocsr()

    return M