"""Update the README from a successful run of the example script."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile


BEGIN = "<!-- BEGINEXAMPLERESULTS -->"
END = "<!-- ENDEXAMPLERESULTS -->"


def _float(value):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("A numerical result is not finite")
    return number


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _record(case):
    steps = case.get("accepted_steps", [])
    counts = case.get("newton_iterations", [])
    corrections = [row for row in case.get("correction_rows", []) if row.get("accepted", True)]
    dt_values = [float(row["dt"]) for row in steps if row.get("dt") is not None]
    residuals = [float(row["linear_relative_residual"]) for row in corrections if row.get("linear_relative_residual") is not None]
    nx = int(case["nx"])
    dry = case.get("problem") == "dry_infiltration"
    h = 2. / (nx - 1) if dry else case.get("h", 1. / (nx - 1))
    return dict(id=str(case["id"]), dimension=int(case["dimension"]), scheme=str(case["scheme"]),
                problem=str(case.get("problem", "")), dataset=case.get("dataset"), time_model=case.get("time_model"),
                nx=nx, nz=int(case["nz"]) if case.get("nz") is not None else None,
                h=_float(h), h_unit="cm" if dry else ("h/L" if case.get("dataset") else "unit domain"),
                dt=_float(case.get("dt")), ell=int(case["ell"]) if case.get("ell") is not None else None,
                dt_min=min(dt_values, default=None), dt_max=max(dt_values, default=None),
                t_final=_float(case.get("t_final")), workers=int(case.get("workers", 1)), relaxation=_float(case.get("relaxation", 1.)),
                error_L2=_float(case.get("error_L2")), error_H1=_float(case.get("error_H1")),
                newton_iterations_mean=_float(case.get("newton_iterations_mean", sum(counts) / len(counts) if counts else 0.)),
                newton_iterations_max=int(case.get("newton_iterations_max", max(counts, default=0))),
                maximum_linear_relative_residual=_float(case.get("maximum_linear_relative_residual", max(residuals, default=None))),
                pressure_min=_float(case.get("pressure_min")), pressure_max=_float(case.get("pressure_max")),
                elapsed_seconds=_float(case.get("wall_seconds", case.get("elapsed_seconds"))),
                newton_tolerance=_float(case.get("newton_tolerance", 1e-8)), linear_solver=case.get("linear_solver"))


def _fraction(value):
    reciprocal = 1. / value
    return f"1/{round(reciprocal)}" if abs(reciprocal - round(reciprocal)) < 1e-9 else f"{value:g}"


def _display(value):
    return "--" if value is None else f"{value:.6e}".replace("e-0", "e-").replace("e+0", "e+")


def _table(cases):
    indexed = any(case.get("ell") is not None for case in cases if case["problem"] == "polynomial_validation")
    rows = ["| Study | Scheme | h | " + ("ℓ | " if indexed else "") + "τ | L² error | H¹ error | Newton max |",
            "| --- | --- | --- | " + ("--- | " if indexed else "") + "--- | --- | --- | --- |"]
    for dimension in (2, 3):
        for study, models in (("spatial", ("stationary",)), ("temporal", ("cubic", "exponential"))):
            for scheme, label in (("implicit", "Implicit"), ("semi_implicit", "Semi-implicit")):
                selected = [case for case in cases if case["dimension"] == dimension and case["scheme"] == scheme
                            and case["problem"] == "polynomial_validation" and case["time_model"] in models]
                if not selected:
                    continue
                case = min(selected, key=lambda item: (item["dt"], item["h"])) if study == "temporal" else min(selected, key=lambda item: (item["h"], item["dt"]))
                index = (str(case["ell"]) if case.get("ell") is not None else "--") + " | " if indexed else ""
                rows.append(f"| {dimension}D {study} | {label} | {_fraction(case['h'])} | {index}{_fraction(case['dt'])} | {_display(case['error_L2'])} | {_display(case['error_H1'])} | {case['newton_iterations_max']} |")
    return "\n".join(rows)


def _sphere_caption(cases):
    spheres = []
    for dataset in ("Braudeau", "Garnier"):
        selected = [case for case in cases if case["problem"] == "wet_slotted_sphere" and case["dataset"] == dataset]
        if selected:
            spheres.append(max(selected, key=lambda case: case["nx"]))
    if not spheres:
        raise ValueError("The README image needs the computed slotted-sphere cases")
    settings = []
    for case in spheres:
        label = f"{case['dataset']}: h/L = {_fraction(1. / (case['nx'] - 1))}, τ = {case['dt']:g} h, T = {case['t_final']:g} h, nx = {case['nx']}"
        if label not in settings:
            settings.append(label)
    return "Native P1 pressure isosurfaces at ψ = −3.29 cm, computed on the tetrahedral mesh at t = 0 and t = T, at true geometric scale. " + "; ".join(settings) + "."


def _block(cases, checks, metadata):
    tolerances = sorted({case["newton_tolerance"] for case in cases})
    tolerance = f"{tolerances[0]:g}".replace("e-0", "e-") if len(tolerances) == 1 else "see the per-case data"
    revision = metadata["source_revision"]
    source = f" Source revision: `{revision[:12]}`." if revision else " Source fingerprints are recorded in the JSON file."
    if metadata.get("parameters_sha256"):
        source += f" Parameter fingerprint: `{metadata['parameters_sha256'][:12]}`."
    settings_text = ""
    temporal = (metadata.get("parameters") or {}).get("temporal")
    if temporal:
        levels = ", ".join(str(level) for level in temporal["ell"])
        settings_text = (f"Temporal validation: **T = {temporal['final_time']:g}**, "
                         f"**h = 1/{temporal['mesh_2d']}** in 2D and **h = 1/{temporal['mesh_3d']}** in 3D; "
                         f"time-step indices **ℓ = {levels}**.\n\n")
    return (BEGIN + "\n## Computed results\n\n"
            f"**{len(cases)} calculations completed · {len(checks)}/{len(checks)} checks passed.** "
            f"Newton stopping tolerance: `{tolerance}`.\n\n"
            f"Last successful run: **{metadata['generated_utc']}**.{source}\n\n"
            f"{settings_text}"
            '<p align="center">\n  <img src="images/slotted_sphere.png" width="850" alt="Computed three-dimensional pressure isosurfaces of the slotted sphere, with initial and final fields">\n</p>\n\n'
            "*" + _sphere_caption(cases) + "*\n\n"
            "The table shows the finest spatial grid and smallest time step of each validation group. H¹ is the full norm.\n\n"
            + _table(cases) + "\n\n"
            f"[All {len(cases)} cases: JSON](data/latest_results.json) · [CSV](data/latest_results.csv)\n\n"
            "To refresh these values locally, run `python test.py --no-show --update-readme`, then commit and push the generated README, image and data files. "
            "Visitor runs in Colab keep their results in their own session and do not publish to this repository.\n"
            + END)


def _replace_block(text, block):
    if BEGIN in text or END in text:
        if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
            raise ValueError("The README result markers are incomplete or duplicated")
        start, end = text.index(BEGIN), text.index(END) + len(END)
        return text[:start] + block + text[end:]
    old = r'<p\b[^>]*>\s*<img\b(?=[^>]*\bsrc=[\"\x27]images/solution_2d\.png[\"\x27])[^>]*>\s*</p>\s*(?:\*Computed pressure field[^\n]*\*\s*)?'
    updated, count = re.subn(old, lambda _match: block + "\n\n", text, count=1)
    if count:
        return updated
    anchor = "## What the test covers"
    if anchor in text:
        return text.replace(anchor, block + "\n\n" + anchor, 1)
    return text.rstrip() + "\n\n" + block + "\n"


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def update_readme(results, checks, images, root, summary_path, parameters=None):
    """Publish only scalar results and the figure from this successful run."""
    root = Path(root).resolve()
    if not results or not checks:
        raise ValueError("No complete calculation and check set is available")
    if any(case.get("status", case.get("run_status")) not in ("complete", "completed", "passed") or case.get("error") for case in results):
        raise ValueError("A calculation did not complete; keep the previous README results")
    if any(check.get("status") != "passed" for check in checks):
        raise ValueError("A numerical check did not pass; keep the previous README results")
    sphere_images = [Path(image) for image in images if Path(image).name == "slotted_sphere.png"]
    if len(sphere_images) != 1 or not sphere_images[0].is_file():
        raise ValueError("The computed slotted_sphere.png is missing or ambiguous")
    image_bytes = sphere_images[0].read_bytes()
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The slotted-sphere image is not a PNG file")
    cases = [_record(case) for case in results]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Calculation identifiers must be unique")
    revision = os.environ.get("BRAUDEAU_SOURCE_REVISION") or os.environ.get("GITHUB_SHA")
    revision = revision.lower() if revision and re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision) else None
    paths = [root / "test.py", root / "parameters.json"] + sorted((root / "source").rglob("*.py"))
    paths = [path for path in paths if path.is_file() and path.name not in ("__init__.py", "readme.py", "figures.py")]
    source_hashes = {path.relative_to(root).as_posix(): _hash(path.read_bytes().replace(b"\r\n", b"\n")) for path in paths}
    check_data = []
    for check in checks:
        entry = dict(name=str(check["name"]), status="passed")
        entry.update({key: _float(check[key]) for key in ("value", "reference", "tolerance") if check.get(key) is not None})
        check_data.append(entry)
    if parameters is not None and not isinstance(parameters, dict):
        raise TypeError("parameters must be the normalized parameter dictionary")
    parameter_json = json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) if parameters is not None else None
    parameter_data = json.loads(parameter_json) if parameter_json is not None else None
    metadata = dict(schema_version=1, status="passed", generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    calculations=len(cases), checks_total=len(checks), checks_passed=len(checks), source_revision=revision,
                    parameters=parameter_data, parameters_sha256=_hash(parameter_json.encode("utf8")) if parameter_json is not None else None,
                    numerical_source_sha256_lf=source_hashes,
                    text_summary_sha256_lf=_hash(Path(summary_path).read_bytes().replace(b"\r\n", b"\n")),
                    image_sha256=_hash(image_bytes), cases=cases, checks=check_data)
    readme = root / "README.md"
    original = readme.read_text(encoding="utf8")
    updated = _replace_block(original, _block(cases, checks, metadata)).replace("\r\n", "\n").replace("\r", "\n")
    json_text = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=list(cases[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(cases)
    if readme.read_text(encoding="utf8") != original:
        raise RuntimeError("The README changed while its results were being prepared")
    _write(root / "data/latest_results.json", json_text.encode("utf8"))
    _write(root / "data/latest_results.csv", csv_text.getvalue().encode("utf8"))
    _write(root / "images/slotted_sphere.png", image_bytes)
    _write(readme, updated.encode("utf8"))
    return readme
