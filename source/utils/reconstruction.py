"""Vertical constitutive display; this does not solve mechanical equilibrium."""
import numpy as np


def vertical_reconstruction(points, initial_pressure, pressure, laws, amplification=1.):
    """Integrate relative stretch along vertical grid edges, with the base fixed.

    Physical pressure is unchanged. Amplification applies only to the displayed
    displacement; the returned displacement is always in true centimetres.
    """
    points = np.asarray(points, dtype=float)
    dimension = points.shape[1]
    if dimension not in (2, 3) or not np.isfinite(amplification) or amplification <= 0:
        raise ValueError("Use a two- or three-dimensional structured grid and a positive display scale")
    axes = [np.unique(points[:, axis]) for axis in range(dimension)]
    shape = tuple(len(axis) for axis in axes)
    order = "C" if dimension == 2 else "F"
    expected = np.column_stack([coordinate.ravel(order=order) for coordinate in np.meshgrid(*axes, indexing="ij")])
    if expected.shape != points.shape or not np.allclose(points, expected, rtol=0, atol=1e-12):
        raise ValueError("Points must follow the native structured-mesh ordering")
    stretch = laws.lois(initial_pressure)["gz"] / laws.lois(pressure)["gz"]
    values = stretch.reshape(shape, order=order)
    increments = np.diff(axes[-1]) * (values[..., :-1] + values[..., 1:]) / 2
    height = np.concatenate((np.zeros((*shape[:-1], 1)), np.cumsum(increments, axis=-1)), axis=-1)
    displacement = height.ravel(order=order) + axes[-1][0] - points[:, -1]
    mapped = points.copy()
    mapped[:, -1] += amplification * displacement
    return mapped, displacement
