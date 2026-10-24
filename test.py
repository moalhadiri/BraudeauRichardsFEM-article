"""Run the small calculations, print their results and save the figures."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parent
from source.utils.configuration import load_parameters, time_step


def examples(parameters, workers):
    """Build the cases from the validated settings; meshes count subdivisions."""
    from source.api import run_validation, run_fourier_validation, run_physical_3d, run_physical_2d
    spatial = parameters["spatial"]
    for dimension, key in ((2, "meshes_2d"), (3, "meshes_3d")):
        for scheme in ("implicit", "semi_implicit"):
            for n in spatial[key]:
                yield f"spatial_{dimension}d_{scheme}_h{n}", run_validation, dict(dimension=dimension, scheme=scheme, nx=n+1, dt=time_step(spatial["ell"]), t_final=spatial["final_time"], workers=workers), spatial["ell"]
    temporal = parameters["temporal"]
    for dimension, key, model in ((2, "mesh_2d", "cubic"), (3, "mesh_3d", "exponential")):
        for scheme in ("implicit", "semi_implicit"):
            for ell in temporal["ell"]:
                dt = time_step(ell)
                name = f"temporal_{dimension}d_{scheme}_ell{ell}_tau{dt:g}"
                yield name, run_validation, dict(dimension=dimension, scheme=scheme, nx=temporal[key]+1, dt=dt, t_final=temporal["final_time"], time_model=model, workers=workers), ell
    fourier = parameters["fourier"]
    yield "fourier_3d_preview", run_fourier_validation, dict(nx=fourier["mesh"]+1, dt=time_step(fourier["ell"]), t_final=fourier["final_time"], workers=workers), fourier["ell"]
    sphere = parameters["sphere"]
    for dataset in ("Braudeau", "Garnier"):
        yield "sphere_" + dataset.lower(), run_physical_3d, dict(dataset=dataset, nx=sphere["mesh"]+1, dt=time_step(sphere["ell"]), t_final=sphere["final_time"], workers=workers), sphere["ell"]
    dry = parameters["dry"]
    yield "dry_2d_preview", run_physical_2d, dict(nx=dry["mesh_x"]+1, nz=dry["mesh_z"]+1, t_final=dry["final_time"], workers=workers), None


def number(value):
    return "--" if value is None else f"{value:.8e}"


def print_result(result, show):
    grid = str(result["nx"]) + (f" x {result['nz']}" if "nz" in result else f" per direction ({result['dimension']}D)")
    show(f"  Scheme: {result['scheme']}; grid: {grid}; relaxation: {result['relaxation']:g}")
    if result.get("ell") is not None:
        show(f"  Time-step level ell: {result['ell']}; tau: {result['dt']:g}")
    show(f"  Nodes: {result['nodes']}; elements: {result['elements']}; final time: {result['t_final']:g}")
    show(f"  L2 error: {number(result.get('error_L2'))}; H1 error: {number(result.get('error_H1'))}")
    for key in ("error_L1", "error_H1_seminorm", "error_Linf_nodes", "error_Linf_quadrature_sample", "theta_min", "theta_max"):
        if key in result:
            show(f"  {key}: {number(result[key])}")
    show(f"  Newton mean/max: {result['newton_iterations_mean']:.4f} / {result['newton_iterations_max']}")
    show(f"  Pressure range: [{result['pressure_min']:.8e}, {result['pressure_max']:.8e}]")
    show(f"  Linear solver: {result['linear_solver']}; elapsed time: {result['wall_seconds']:.3f} s")
    for key in ("parameters", "controller"):
        if key in result:
            show(f"  {key}: " + "; ".join(f"{name}={value}" for name, value in result[key].items()))
    for key in ("rejected_attempts_count", "free_nodes_visited_all_five_branches"):
        if key in result:
            show(f"  {key}: {result[key]}")
    show("  Step        time          dt   Newton       last r_j       ||R||_2")
    for step in result["accepted_steps"]:
        show(f"  {step['step']:4d}  {step['time']:10.5g}  {step['dt']:10.5g}  {step['newton_iterations']:7d}  {step['final_relative_correction']:13.5e}  {step['nonlinear_residual_l2']:13.5e}")
    show("  Correction details: step, j, r_j, true linear relative residual")
    for row in result["correction_rows"]:
        accepted = "" if row.get("accepted", True) else " (rejected time-step trial)"
        show(f"    {row['step']:4d} {row['iteration']:3d}  {row['relative_correction']:.8e}  {row['linear_relative_residual']:.8e}{accepted}")


def print_check(check, show):
    detail = ""
    for key in ("value", "reference", "tolerance"):
        if key in check:
            detail += f"; {key}={number(check[key])}"
    show(f"  CHECK {check['status'].upper()}: {check['name']}{detail}")
    if check["status"] == "failed" and check.get("message"):
        show("    " + check["message"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="number of element-assembly workers")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="folder for text and PNG files")
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening windows")
    parser.add_argument("--update-readme", action="store_true", help="refresh the README and its example data after a successful run")
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters.json", help="JSON file containing meshes, final times and time-step levels")
    parser.add_argument("--check-parameters", action="store_true", help="validate the settings and print time steps without running calculations")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        parameters = load_parameters(args.parameters)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.check_parameters:
        time_steps = {name: time_step(parameters[name]["ell"]) for name in ("spatial", "fourier", "sphere")}
        time_steps["temporal"] = {str(ell): time_step(ell) for ell in parameters["temporal"]["ell"]}
        time_steps["dry"] = "adaptive"
        print(json.dumps(dict(parameters=parameters, time_steps=time_steps), indent=2, allow_nan=False))
        return 0
    try:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from source.utils.figures import save_figures
        from source.utils.verification import numerical_checks, case_checks
    except ImportError as error:
        raise SystemExit(f"Missing dependency: {error}. Run: python -m pip install -r requirements.txt") from error
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    results, checks, failures = [], [], []
    begin = time.perf_counter()
    with (folder / "results.txt").open("w", encoding="utf8") as text:
        def show(message=""):
            print(message, flush=True)
            text.write(message + "\n")
            text.flush()
        show("BRAUDEAU-RICHARDS: SMALL NUMERICAL TESTS")
        show(f"Assembly workers: {args.workers}")
        show("Temporal levels and time steps: " + "; ".join(f"ell={ell}, tau={time_step(ell):g}" for ell in parameters["temporal"]["ell"]))
        show("Newton tolerance: 1e-8; maximum corrections per step: 200.")
        show("GMRES tolerance: 1e-10; checked true residual: at most 1.05e-10.")
        show("The dry 2D calculation uses a sparse direct solve and adaptive time steps.")
        show("r_j is the raw Newton correction divided by the relaxed field, in the consistent-mass norm.")
        show("Physical errors without an exact reference are displayed as --.")
        checks.extend(numerical_checks())
        for check in checks:
            print_check(check, show)
        cases = list(examples(parameters, args.workers))
        for index, (name, solve, kwargs, ell) in enumerate(cases, 1):
            show(f"\n[{index}/{len(cases)}] {name}")
            start = time.perf_counter()
            try:
                result = solve(**kwargs)
            except Exception as error:
                failures.append(name)
                show(f"  FAILED: {type(error).__name__}: {error}")
                continue
            result.update(id=name, ell=ell, wall_seconds=time.perf_counter() - start)
            results.append(result)
            print_result(result, show)
            try:
                current_checks = case_checks(result, ROOT / "data")
            except Exception as error:
                current_checks = [dict(name=name + " verification", status="failed", message=str(error))]
            checks.extend(current_checks)
            for check in current_checks:
                print_check(check, show)
        show("\nFINAL RESULTS")
        show("Case                                           L2 error       H1 error    Newton max")
        for result in results:
            show(f"{result['id']:<43} {number(result.get('error_L2')):>14} {number(result.get('error_H1')):>14} {result['newton_iterations_max']:8d}")
        try:
            images = save_figures(results, folder)
        except Exception as error:
            images = []
            failures.append("figure output")
            show(f"Figure output FAILED: {type(error).__name__}: {error}")
        failed_checks = [check for check in checks if check["status"] == "failed"]
        show(f"\nCalculations completed: {len(results)}/{len(cases)}")
        show(f"Checks passed: {len(checks) - len(failed_checks)}/{len(checks)}")
        show(f"Elapsed time: {time.perf_counter() - begin:.2f} s")
        for name in failures:
            show("FAILED calculation: " + name)
        for check in failed_checks:
            show("FAILED check: " + check["name"])
        show(f"Text output: {folder / 'results.txt'}")
        for image in images:
            show(f"Image: {image}")
        if not failures and not failed_checks:
            show("ALL CALCULATIONS AND CHECKS PASSED.")
    if args.update_readme and not failures and not failed_checks:
        try:
            from source.utils.readme import update_readme
            update_readme(results, checks, images, ROOT, folder / "results.txt", parameters=parameters)
        except Exception as error:
            failures.append("README update")
            print(f"README update FAILED: {type(error).__name__}: {error}", file=sys.stderr)
    if args.no_show:
        plt.close("all")
    elif images:
        plt.show()
    return 1 if failures or failed_checks else 0


if __name__ == "__main__":
    raise SystemExit(main())
