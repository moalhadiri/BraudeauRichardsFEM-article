"""Intersect the native tetrahedral P1 field with a fixed pressure level."""
import numpy as np


def extract_isosurface(points, tetrahedra, values, level):
    """Return triangles with shape ``(nfaces, 3, 3)`` at one fixed scalar level.

    ``points`` is (N, 3), ``tetrahedra`` is zero-based (M, 4), and ``values``
    contains N nodal scalar values for one state. The affine field is intersected
    directly inside each tetrahedron: 1/3 splits give triangles, and 2/2 splits
    give planar quadrilaterals split into two triangles.

    Exact equalities are retained as mesh vertices. A face whose three vertices
    equal the level is included once, even when shared by two tetrahedra.
    Point/edge intersections and degenerate triangles are discarded. Cells with
    four equal-level vertices are omitted: their level set is a volume and has
    no unique surface triangulation. Distinct mesh nodes must identify distinct
    geometric vertices for shared-face deduplication.

    Faces point toward increasing values in the contributing tetrahedron. At a
    face that is a local extremum, this direction is chosen by the first incident
    tetrahedron. Near-collinear triangles are removed only at floating-point
    angular precision, without an absolute area cutoff.
    """
    points = np.asarray(points, dtype=float)
    tetrahedra = np.asarray(tetrahedra)
    values = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must have finite shape (N, 3)")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (M, 4)")
    if not np.issubdtype(tetrahedra.dtype, np.integer):
        raise ValueError("tetrahedra must contain integer node indices")
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError("values must contain one finite scalar per node")
    if np.ndim(level) != 0 or not np.isfinite(level):
        raise ValueError("level must be a finite scalar")
    level = float(level)
    empty = np.empty((0, 3, 3), dtype=float)
    if not len(tetrahedra):
        return empty
    if tetrahedra.min() < 0 or tetrahedra.max() >= len(points):
        raise ValueError("tetrahedral connectivity is out of bounds")

    nodal_values = values[tetrahedra]
    above = nodal_values > level
    number_above = above.sum(axis=1)
    edges_by_case, directions_by_case = [], []

    # Fixed-size case loops; all tetrahedra in each case are processed together.
    for count in (1, 2, 3):
        mask = number_above == count
        if not mask.any():
            continue
        order = np.argsort(~above[mask], axis=1, kind="stable")
        nodes = np.take_along_axis(tetrahedra[mask], order, axis=1)
        direction = points[nodes[:, 0]] - points[nodes[:, -1]]
        if count == 1:
            first = np.broadcast_to(nodes[:, :1], (len(nodes), 3))
            edges = np.stack((first, nodes[:, 1:]), axis=-1)
        elif count == 3:
            first = np.broadcast_to(nodes[:, 3:], (len(nodes), 3))
            edges = np.stack((first, nodes[:, :3]), axis=-1)
        else:
            # Cyclic quad boundary: a-c, a-d, b-d, b-c.
            first = nodes[:, [0, 0, 1, 1]]
            second = nodes[:, [2, 3, 3, 2]]
            quad = np.stack((first, second), axis=-1)
            edges = quad[:, [[0, 1, 2], [0, 2, 3]], :].reshape(-1, 3, 2)
            direction = np.repeat(direction, 2, axis=0)
        edges_by_case.append(edges)
        directions_by_case.append(direction)

    # The <= classification above emits an exact nodal face from the positive
    # side. Include its negative-side counterpart too, then deduplicate globally.
    on_level = nodal_values == level
    mask = (on_level.sum(axis=1) == 3) & (number_above == 0)
    if mask.any():
        order = np.argsort(~on_level[mask], axis=1, kind="stable")
        nodes = np.take_along_axis(tetrahedra[mask], order, axis=1)
        edges_by_case.append(np.stack((nodes[:, :3], nodes[:, :3]), axis=-1))
        directions_by_case.append(points[nodes[:, 0]] - points[nodes[:, 3]])
    if not edges_by_case:
        return empty

    keys = np.sort(np.concatenate(edges_by_case), axis=-1)
    directions = np.concatenate(directions_by_case)
    # Canonical keys distinguish crossing edges, but all edges ending exactly at
    # the same equal-level node must share the node key (node, node).
    first_on = values[keys[..., 0]] == level
    second_on = values[keys[..., 1]] == level
    equal_node = np.where(first_on, keys[..., 0], keys[..., 1])
    keys = np.where((first_on | second_on)[..., None], equal_node[..., None], keys)
    lo, hi = keys[..., 0], keys[..., 1]
    denominator = values[hi] - values[lo]
    fraction = np.divide(level - values[lo], denominator,
                         out=np.zeros_like(denominator), where=denominator != 0)
    triangles = points[lo] + fraction[..., None] * (points[hi] - points[lo])

    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    norm_squared = np.einsum("ij,ij->i", normals, normals)
    edge_product = (np.einsum("ij,ij->i", edge_a, edge_a)
                    * np.einsum("ij,ij->i", edge_b, edge_b))
    keep = norm_squared > (32 * np.finfo(float).eps) ** 2 * edge_product
    triangles, keys, normals, directions = (array[keep] for array in
                                            (triangles, keys, normals, directions))
    if not len(triangles):
        return empty

    # Sort the three vertex keys per face; opposite orientations then have the
    # same key. This removes duplicate coincident faces at exact nodal levels.
    order = np.lexsort((keys[..., 1], keys[..., 0]), axis=1)
    canonical = np.take_along_axis(keys, order[..., None], axis=1).reshape(-1, 6)
    _, unique = np.unique(canonical, axis=0, return_index=True)
    unique.sort()
    triangles, normals, directions = (array[unique] for array in
                                      (triangles, normals, directions))
    reverse = np.einsum("ij,ij->i", normals, directions) < 0
    triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    return np.ascontiguousarray(triangles)
