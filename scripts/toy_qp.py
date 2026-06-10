"""Toy QP from the paper (Fig. 3): solve, validate, and visualize.

Problem:
    min  1/2 (x1^2 + x2^2)
    s.t. x1 + x2 >=  0.65
                x2 >= -0.1
        -x1       >= -0.85
              -x2 >= -0.8

The unconstrained minimum is the origin, but x1 + x2 >= 0.65 cuts it off, so
the optimum lies on that line at x* = (0.325, 0.325). The other three
constraints are slack (inactive). This script confirms that and plots both the
feasible region and the convergence of the implicit interior-point solver.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from implicit_ipm_qp.qp import QP
from implicit_ipm_qp.solvers.implicit import SolverResult, solve_implicit
from implicit_ipm_qp.utils.logging import configure_logging, get_logger

_log = get_logger(__name__)

# Where to write the figure.
FIGURE_PATH = Path(__file__).resolve().parent.parent / "figures" / "toy_qp.png"


def build_toy_qp() -> QP:
    """Construct the Fig. 3 toy QP (Q = I, q = 0, four inequalities, no equalities)."""
    q_mat = np.eye(2)
    q_vec = np.zeros(2)
    a_mat = np.array(
        [
            [1.0, 1.0],  # x1 + x2 >= 0.65
            [0.0, 1.0],  #      x2 >= -0.1
            [-1.0, 0.0],  #     -x1 >= -0.85
            [0.0, -1.0],  #     -x2 >= -0.8
        ]
    )
    b_vec = np.array([0.65, -0.1, -0.85, -0.8])
    c_mat = np.zeros((0, 2))
    d_vec = np.zeros(0)
    return QP(Q=q_mat, q=q_vec, A=a_mat, b=b_vec, C=c_mat, d=d_vec)


def validate(result: SolverResult) -> None:
    """Assert the known optimum and the expected active/inactive constraint set."""
    expected = np.array([0.325, 0.325])
    if not result.converged:
        raise RuntimeError("solver did not converge on the toy QP")
    np.testing.assert_allclose(result.x, expected, rtol=1e-3, atol=1e-3)

    # A constraint is active iff its slack s_i ~ 0. Only constraint 0 should be.
    active = result.s < 1e-3
    expected_active = np.array([True, False, False, False])
    if not np.array_equal(active, expected_active):
        raise RuntimeError(f"unexpected active set: {active.tolist()}")

    _log.info(
        "toy_qp_validated",
        x=result.x.tolist(),
        active_constraints=np.flatnonzero(active).tolist(),
        iterations=result.iterations,
    )


def plot(qp: QP, result: SolverResult, path: Path) -> None:
    """Two panels: the feasible region with the solution, and convergence curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_geom, ax_conv) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- left: feasible region, cost level curves, and constraint lines -
    grid = np.linspace(-0.3, 1.1, 400)
    xx, yy = np.meshgrid(grid, grid)

    # Objective level curves: f(x) = 1/2 xᵀQx + qᵀx (here concentric circles).
    cost = (
        0.5 * (qp.Q[0, 0] * xx**2 + (qp.Q[0, 1] + qp.Q[1, 0]) * xx * yy + qp.Q[1, 1] * yy**2)
        + qp.q[0] * xx
        + qp.q[1] * yy
    )
    cs = ax_geom.contour(xx, yy, cost, levels=12, colors="#999999", linewidths=0.6, alpha=0.8)
    ax_geom.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    # Feasible region (shaded), drawn over the level curves.
    feasible = np.ones_like(xx, dtype=bool)
    for i in range(qp.m):
        feasible &= (qp.A[i, 0] * xx + qp.A[i, 1] * yy) >= qp.b[i]
    ax_geom.contourf(xx, yy, feasible, levels=[0.5, 1.5], colors=["#cfe8ff"], alpha=0.45)

    for i in range(qp.m):
        a0, a1 = qp.A[i]
        if abs(a1) > 1e-12:
            ax_geom.plot(grid, (qp.b[i] - a0 * grid) / a1, lw=1, color="#5a6b7b")
        else:
            ax_geom.axvline(qp.b[i] / a0, lw=1, color="#5a6b7b")

    ax_geom.scatter(*result.x, color="#d62728", zorder=5, s=70, label="solution")
    ax_geom.scatter(0, 0, color="#444", marker="x", s=50, label="unconstrained min")
    ax_geom.set_xlim(-0.3, 1.1)
    ax_geom.set_ylim(-0.3, 1.1)
    ax_geom.set_aspect("equal")
    ax_geom.set_xlabel("x1")
    ax_geom.set_ylabel("x2")
    ax_geom.set_title("Feasible region and solution")
    ax_geom.legend(loc="upper right", fontsize=8)

    # --- right: convergence (log scale) ---------------------------------
    iters = range(1, len(result.gap_history) + 1)
    ax_conv.semilogy(iters, result.gap_history, marker="o", ms=3, label="duality gap")
    ax_conv.semilogy(iters, result.stat_history, marker="s", ms=3, label="||r_stat||")
    ax_conv.set_xlabel("iteration")
    ax_conv.set_ylabel("residual / gap")
    ax_conv.set_title("Implicit IPM convergence")
    ax_conv.grid(True, which="both", ls=":", alpha=0.5)
    ax_conv.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _log.info("figure_saved", path=str(path))


def main() -> None:
    configure_logging()
    qp = build_toy_qp()
    result = solve_implicit(qp)
    validate(result)
    plot(qp, result, FIGURE_PATH)
    _log.info("done", solution=result.x.tolist())


if __name__ == "__main__":
    main()
