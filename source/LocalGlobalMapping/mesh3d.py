"""Uniform box mesh with six tetrahedra in each Cartesian cell."""
import numpy as np

def kpde3dumsh(ax, bx, ay, by, az, bz, nx, ny, nz):
    """Return nodes, tetrahedra and node indices on each pair of boundary faces."""

    np_nodes = nx * ny * nz
    nq = (nx - 1) * (ny - 1) * (nz - 1)
    nt = 6 * nq

    hx = (bx - ax) / (nx - 1)
    hy = (by - ay) / (ny - 1)
    hz = (bz - az) / (nz - 1)

    x = ax + hx * np.arange(nx)
    y = ay + hy * np.arange(ny)
    z = az + hz * np.arange(nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    p = np.column_stack((X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")))

    ip = np.arange(np_nodes, dtype=int)

    ijk = ip.reshape((nx, ny, nz), order="F")

    b1 = ijk[:, 0, :]
    ib1 = b1.ravel(order="F")

    b2 = ijk[nx - 1, :, :]
    ib2 = b2.ravel(order="F")

    b3 = ijk[:, ny - 1, :]
    ib3 = b3.ravel(order="F")

    b4 = ijk[0, :, :]
    ib4 = b4.ravel(order="F")

    b5 = ijk[:, :, 0]
    ib5 = b5.ravel(order="F")

    b6 = ijk[:, :, nz - 1]
    ib6 = b6.ravel(order="F")

    ib236 = np.union1d(ib2, np.union1d(ib3, ib6))
    ip1 = np.setdiff1d(ip, ib236)

    ib346 = np.union1d(ib3, np.union1d(ib4, ib6))
    ip2 = np.setdiff1d(ip, ib346)

    ib345 = np.union1d(ib3, np.union1d(ib4, ib5))
    ip3 = np.setdiff1d(ip, ib345)

    ib235 = np.union1d(ib2, np.union1d(ib3, ib5))
    ip4 = np.setdiff1d(ip, ib235)

    ib126 = np.union1d(ib1, np.union1d(ib2, ib6))
    ip5 = np.setdiff1d(ip, ib126)

    ib146 = np.union1d(ib1, np.union1d(ib4, ib6))
    ip6 = np.setdiff1d(ip, ib146)

    ib145 = np.union1d(ib1, np.union1d(ib4, ib5))
    ip7 = np.setdiff1d(ip, ib145)

    ib125 = np.union1d(ib1, np.union1d(ib2, ib5))
    ip8 = np.setdiff1d(ip, ib125)

    t = np.zeros((nt, 4), dtype=int)

    iq1 = np.arange(nq)
    iq2 = iq1 + nq
    iq3 = iq2 + nq
    iq4 = iq3 + nq
    iq5 = iq4 + nq
    iq6 = iq5 + nq

    t[iq1, :] = np.column_stack((ip1, ip2, ip6, ip7))
    t[iq2, :] = np.column_stack((ip1, ip6, ip5, ip7))
    t[iq3, :] = np.column_stack((ip1, ip2, ip7, ip3))
    t[iq4, :] = np.column_stack((ip1, ip4, ip3, ip7))
    t[iq5, :] = np.column_stack((ip1, ip5, ip8, ip7))
    t[iq6, :] = np.column_stack((ip1, ip4, ip7, ip8))

    pbx = np.column_stack((ib4, ib2))
    pby = np.column_stack((ib1, ib3))
    pbz = np.column_stack((ib5, ib6))

    return p, t, pbx, pby, pbz