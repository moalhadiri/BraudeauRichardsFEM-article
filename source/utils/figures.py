"""Plot the computed fields and errors from the small test."""

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator
from matplotlib.tri import Triangulation
import numpy as np

from .isosurface import extract_isosurface


SCHEMES = (("implicit", "Implicit", "o"), ("semi_implicit", "Semi-implicit", "s"))


def _error(case, norm):
    value = case.get("error_" + norm, (case.get("errors") or {}).get(norm))
    return float(value) if value is not None else np.nan


def _h(case):
    return float(case.get("h", 1. / (int(case["nx"]) - 1)))


def _validation(case):
    return not case.get("dataset") and case.get("problem", "polynomial_validation") == "polynomial_validation"


def _errors(axes, cases, norm, temporal=False):
    colours = plt.colormaps["viridis"]([.2, .7])
    for (scheme, label, marker), colour in zip(SCHEMES, colours):
        selected = [case for case in cases if case.get("scheme") == scheme and np.isfinite(_error(case, norm)) and _error(case, norm) > 0]
        groups = sorted({int(case["nx"]) if temporal else float(case.get("dt", 0)) for case in selected})
        for group in groups:
            entries = [case for case in selected if (int(case["nx"]) if temporal else float(case.get("dt", 0))) == group]
            entries.sort(key=lambda case: float(case["dt"]) if temporal else _h(case), reverse=True)
            x = [float(case["dt"]) if temporal else _h(case) for case in entries]
            suffix = (f", h=1/{group - 1}" if temporal else f", tau={group:g}") if len(groups) > 1 else ""
            axes.loglog(x, [_error(case, norm) for case in entries], marker=marker, linewidth=2, markersize=8 if scheme == "implicit" else 6, markerfacecolor="none" if scheme == "implicit" else colour, linestyle="-" if scheme == "implicit" else "--", color=colour, label=label + suffix)
    axes.set(xlabel=r"$\tau$" if temporal else "$h$", ylabel="$" + norm[0] + "^" + norm[1] + "$ error")
    points = sorted({float(case["dt"]) if temporal else _h(case) for case in cases})
    axes.set_xticks(points, [str(Fraction(value).limit_denominator(10000)) if temporal else f"1/{round(1 / value)}" for value in points])
    axes.xaxis.set_minor_locator(NullLocator())
    axes.grid(True, which="major", color=".88", linewidth=.7)
    if axes.lines:
        axes.legend(fontsize=10, frameon=False)


def save_figures(results, folder):
    """Save PNG files and leave the figures open for plt.show()."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    results = [case for case in results if case.get("status", case.get("run_status", "complete")) in ("complete", "completed", "passed", "converged")]
    files = []

    def save(fig, name):
        path = folder / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        files.append(path)

    style = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "lines.linewidth": 2, "text.usetex": False}
    with plt.rc_context(style):
        for dimension in (2, 3):
            cases = [case for case in results if case.get("dimension") == dimension and _validation(case) and case.get("time_model") == "stationary"]
            if cases:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
                for axis, norm in zip(axes, ("L2", "H1")):
                    _errors(axis, cases, norm)
                fig.suptitle(f"{dimension}D spatial errors")
                save(fig, f"spatial_errors_{dimension}d.png")

        temporal = [case for case in results if _validation(case) and case.get("time_model") in ("cubic", "exponential")]
        if temporal:
            dimensions = sorted({int(case["dimension"]) for case in temporal})
            fig, axes = plt.subplots(len(dimensions), 2, figsize=(11, 4.2 * len(dimensions)), squeeze=False, layout="constrained")
            for row, dimension in enumerate(dimensions):
                cases = [case for case in temporal if case["dimension"] == dimension]
                for axis, norm in zip(axes[row], ("L2", "H1")):
                    _errors(axis, cases, norm, temporal=True)
                    axis.set_title(f"{dimension}D")
            fig.suptitle("Temporal errors")
            save(fig, "temporal_errors.png")

        surfaces = [case for case in results if case.get("dimension") == 2 and _validation(case) and case.get("scheme") == "implicit" and case.get("time_model") == "stationary" and all(key in case for key in ("p", "t", "final"))]
        if surfaces:
            case = max(surfaces, key=lambda item: int(item["nx"]))
            p, t, values = np.asarray(case["p"]), np.asarray(case["t"]), np.asarray(case["final"])
            fig = plt.figure(figsize=(8.2, 6.4), layout="constrained")
            axes = fig.add_subplot(projection="3d")
            surface = axes.plot_trisurf(Triangulation(p[:, 0], p[:, 1], t), values, cmap="viridis", linewidth=.2, edgecolor=".3")
            axes.set(xlabel="$x$", ylabel="$z$", zlabel=r"$\psi_h$", title=f"2D implicit solution, h=1/{int(case['nx']) - 1}, T={case['t_final']:g}")
            axes.view_init(elev=25, azim=-125)
            fig.colorbar(surface, ax=axes, shrink=.65, pad=.1, label=r"$\psi_h$")
            save(fig, "solution_2d.png")

        spheres = []
        for dataset in ("Braudeau", "Garnier"):
            cases = [case for case in results if case.get("dimension") == 3 and case.get("dataset") == dataset and all(key in case for key in ("p", "t", "initial", "final"))]
            if cases:
                spheres.append(max(cases, key=lambda item: int(item["nx"])))
        if spheres:
            level = -3.29
            colour = plt.colormaps["viridis"](Normalize(-5., -1.)(level))
            fig = plt.figure(figsize=(11.4, 4.7 * len(spheres)))
            for row, case in enumerate(spheres):
                length = float(case.get("length_cm", 2.75))
                p, t = np.asarray(case["p"]), np.asarray(case["t"])
                spacing = str(Fraction(float(case.get("h", 1. / (int(case["nx"]) - 1)))).limit_denominator(10000))
                for column, key in enumerate(("initial", "final")):
                    axis = fig.add_subplot(len(spheres), 2, 2 * row + column + 1, projection="3d")
                    triangles = extract_isosurface(p, t, np.asarray(case[key]), level)
                    if len(triangles):
                        surface = Poly3DCollection(triangles / length, facecolors=colour, edgecolors="black", linewidths=.16, alpha=1.)
                        axis.add_collection3d(surface)
                    else:
                        axis.text2D(.15, .5, "This pressure level is absent", transform=axis.transAxes)
                    time = 0. if key == "initial" else float(case["t_final"])
                    axis.set(xlim=(0., 1.), ylim=(0., 1.), zlim=(0., 1.), xlabel="$x/L$", ylabel="$y/L$", zlabel="$z/L$")
                    axis.set_xticks((0., .5, 1.))
                    axis.set_yticks((0., .5, 1.))
                    axis.set_zticks((0., .5, 1.))
                    axis.tick_params(labelsize=11, pad=0)
                    axis.tick_params(axis="x", pad=5)
                    for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
                        coordinate.set_major_formatter(FormatStrFormatter("%g"))
                    axis.set_box_aspect((1, 1, 1))
                    axis.view_init(elev=10, azim=10)
                    axis.grid(False)
                    axis.set_title(f"{case['dataset']}, t={time:g} h\n" + rf"$h/L={spacing}$, $\tau={case['dt']:g}$ h, $T={case['t_final']:g}$ h", fontsize=12, pad=9)
            fig.suptitle(r"Slotted wet sphere: native $P_1$ pressure isosurface $\psi=-3.29$ cm", fontsize=15, y=.985)
            fig.subplots_adjust(left=.035, right=.965, bottom=.035, top=.89, wspace=.10, hspace=.28)
            save(fig, "slotted_sphere.png")

        dry = [case for case in results if case.get("dimension") == 2 and case.get("problem") == "dry_infiltration" and all(key in case for key in ("p", "t", "initial", "final"))]
        if dry:
            case = max(dry, key=lambda item: int(item["nx"]))
            p, t = np.asarray(case["p"]), np.asarray(case["t"])
            triangulation = Triangulation(p[:, 0], p[:, 1], t)
            low, high = min(float(np.min(case[key])) for key in ("initial", "final")), max(float(np.max(case[key])) for key in ("initial", "final"))
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
            for axis, key in zip(axes, ("initial", "final")):
                time = 0. if key == "initial" else float(case["t_final"])
                image = axis.tripcolor(triangulation, np.asarray(case[key]), shading="gouraud", cmap="viridis", vmin=low, vmax=high)
                axis.set(xlabel="x (cm)", ylabel="z (cm)", aspect="equal", title=f"t={time:g} h")
            fig.colorbar(image, ax=axes, shrink=.8, label=r"Pressure $\psi$ (cm)")
            fig.suptitle("2D dry infiltration")
            save(fig, "dry_infiltration.png")

        newton = [case for case in results if _validation(case) and case.get("time_model") == "stationary" and case.get("correction_rows")]
        if newton:
            dimensions = sorted({int(case["dimension"]) for case in newton})
            fig, axes = plt.subplots(1, len(dimensions), figsize=(6 * len(dimensions), 6.1), squeeze=False)
            for axis, dimension in zip(axes[0], dimensions):
                cases = sorted([case for case in newton if case["dimension"] == dimension], key=lambda item: (item.get("scheme", ""), int(item["nx"])))
                colours = plt.colormaps["viridis"](np.linspace(.1, .85, max(1, len(cases))))
                for case, colour in zip(cases, colours):
                    rows = [row for row in case["correction_rows"] if row.get("accepted", True)]
                    step = min((row.get("step", 1) for row in rows), default=1)
                    rows = [row for row in rows if row.get("step", 1) == step]
                    values = [row.get("relative_correction", np.nan) for row in rows]
                    label = "Implicit" if case.get("scheme") == "implicit" else "Semi-implicit"
                    axis.semilogy(np.arange(1, len(values) + 1), values, marker="o", linewidth=2, markersize=5, color=colour, label=f"{label}, h=1/{int(case['nx']) - 1}")
                axis.axhline(1e-8, color=".45", linewidth=1, linestyle="--")
                axis.set(xlabel="Newton iteration", ylabel="Relative correction", title=f"{dimension}D, first time step")
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.grid(axis="y", color=".88", linewidth=.7)
                axis.legend(loc="upper center", bbox_to_anchor=(.5, -.18), fontsize=9, ncol=2, frameon=False)
            fig.subplots_adjust(left=.09, right=.98, bottom=.28, top=.9, wspace=.28)
            save(fig, "newton_corrections.png")
    return files
