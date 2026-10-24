"""Read the example settings and check them before any calculation starts."""
import json
import math


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}; booleans are not accepted")


def time_step(ell):
    """ell selects the time step; it does not change the constitutive exponent."""
    _integer(ell, "ell", 1)
    try:
        value = 0.1 * 2 ** (1 - ell)
    except OverflowError as error:
        raise ValueError("ell produces a time step that cannot be represented") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("ell produces a time step that is too small to represent")
    return value


def _keys(values, expected, name):
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a JSON object")
    missing = sorted(set(expected) - set(values))
    extra = sorted(set(values) - set(expected))
    if missing:
        raise ValueError(f"Missing parameter in {name}: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unknown parameter in {name}: {', '.join(extra)}")


def _unique_integers(values, name, minimum):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list of integers")
    for index, value in enumerate(values):
        _integer(value, f"{name}[{index}]", minimum)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")


def _final_time(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite positive number")


def _time_grid(final_time, ell, name):
    dt = time_step(ell)
    ratio = final_time / dt
    if not math.isfinite(ratio):
        raise ValueError(f"{name}.final_time / tau is too large to represent")
    steps = round(ratio)
    if steps < 1 or not math.isclose(steps * dt, final_time, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{name}.final_time={final_time:g} must be a positive integer multiple of tau={dt:g} for ell={ell}")


def _object(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"Duplicate JSON key: {key}")
        values[key] = value
    return values


def load_parameters(path):
    """Load the specified file; missing files and invalid settings are not replaced."""
    try:
        with open(path, encoding="utf-8-sig") as stream:
            values = json.load(stream, object_pairs_hook=_object)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid parameters JSON at line {error.lineno}, column {error.colno}: {error.msg}") from error
    schema = {
        "spatial": ("meshes_2d", "meshes_3d", "ell", "final_time"),
        "temporal": ("mesh_2d", "mesh_3d", "ell", "final_time"),
        "fourier": ("mesh", "ell", "final_time"),
        "sphere": ("mesh", "ell", "final_time"),
        "dry": ("mesh_x", "mesh_z", "final_time"),
    }
    _keys(values, schema, "parameters")
    for name, fields in schema.items():
        section = values[name]
        _keys(section, fields, name)
        _final_time(section["final_time"], name + ".final_time")
        for key in fields:
            if key.startswith("meshes_"):
                _unique_integers(section[key], name + "." + key, 2)
            elif key.startswith("mesh"):
                _integer(section[key], name + "." + key, 2)
        if name == "dry":
            continue
        if name == "temporal":
            _unique_integers(section["ell"], name + ".ell", 1)
            levels = section["ell"]
        else:
            _integer(section["ell"], name + ".ell", 1)
            levels = (section["ell"],)
        for ell in levels:
            _time_grid(section["final_time"], ell, name)
    return values


def automatic_parameters(mesh_2d, mesh_3d, final_time, levels):
    """Use the common settings for all examples and three spatial grids."""
    _integer(mesh_2d, "mesh_2d", 4)
    _integer(mesh_3d, "mesh_3d", 4)
    _final_time(final_time, "final_time")
    _unique_integers(levels, "time_levels", 1)
    for level in levels:
        _time_grid(final_time, level, "automatic")
    finest = max(levels)
    meshes_2d = [max(2, mesh_2d // 4), max(3, mesh_2d // 2), mesh_2d]
    meshes_3d = [max(2, mesh_3d // 4), max(3, mesh_3d // 2), mesh_3d]
    return {
        "spatial": dict(meshes_2d=meshes_2d, meshes_3d=meshes_3d, ell=finest, final_time=final_time),
        "temporal": dict(mesh_2d=mesh_2d, mesh_3d=mesh_3d, ell=list(levels), final_time=final_time),
        "fourier": dict(mesh=mesh_3d, ell=finest, final_time=final_time),
        "sphere": dict(mesh=mesh_3d, ell=finest, final_time=final_time),
        "dry": dict(mesh_x=2 * mesh_2d, mesh_z=mesh_2d, final_time=final_time),
    }
