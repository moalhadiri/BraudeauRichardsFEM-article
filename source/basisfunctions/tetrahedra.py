"""Tetrahedron volumes and gradients of the four linear basis functions."""
import numpy as np

def kpde3dgphi(p, t):
    """Return cell volumes and the gradients of the four P1 basis functions."""

    it1 = t[:, 0]
    it2 = t[:, 1]
    it3 = t[:, 2]
    it4 = t[:, 3]

    x1 = p[it1, 0]
    x2 = p[it2, 0]
    x3 = p[it3, 0]
    x4 = p[it4, 0]

    y1 = p[it1, 1]
    y2 = p[it2, 1]
    y3 = p[it3, 1]
    y4 = p[it4, 1]

    z1 = p[it1, 2]
    z2 = p[it2, 2]
    z3 = p[it3, 2]
    z4 = p[it4, 2]

    J11 = x2 - x1
    J12 = y2 - y1
    J13 = z2 - z1

    J21 = x3 - x1
    J22 = y3 - y1
    J23 = z3 - z1

    J31 = x4 - x1
    J32 = y4 - y1
    J33 = z4 - z1

    det = (
        J11 * (J22 * J33 - J32 * J23)
        + J12 * (J31 * J23 - J21 * J33)
        + J13 * (J21 * J32 - J31 * J22)
    )

    vol = np.abs(det) / 6.0

    C11 = (J22 * J33 - J32 * J23) / det
    C12 = (J13 * J32 - J12 * J33) / det
    C13 = (J12 * J23 - J13 * J22) / det

    C21 = (J31 * J23 - J21 * J33) / det
    C22 = (J11 * J33 - J13 * J31) / det
    C23 = (J21 * J13 - J23 * J11) / det

    C31 = (J21 * J32 - J31 * J22) / det
    C32 = (J12 * J31 - J32 * J11) / det
    C33 = (J11 * J22 - J12 * J21) / det

    g2 = np.column_stack((C11, C21, C31))
    g3 = np.column_stack((C12, C22, C32))
    g4 = np.column_stack((C13, C23, C33))

    g1 = -g2 - g3 - g4

    dphidz = np.concatenate(
        (
            C13,
            C23,
            C33,
            -(C13 + C23 + C33),
        )
    )

    return vol, g1, g2, g3, g4, dphidz