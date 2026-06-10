"""Implicit interior-point primal-dual solver for convex QP.

Solves   min  1/2 xᵀQx + qᵀx   s.t.   Ax >= b,  Cx = d,   Q symmetric PSD.

The "implicit" method (paper arXiv:2604.00364) replaces the complementarity
condition lambda ⊙ s = 0 with the retraction map b_mu:

    lambda = b_mu(v),    s = b_mu(-v),    v in R^m.

Because b_mu(v) * b_mu(-v) = mu and db_mu(v) + db_mu(-v) = 1 (Definition 1),
the reduced Newton block that would otherwise be -Lambda^{-1} S (entries
-s_i/lambda_i, unbounded as mu -> 0) becomes -B_mu(-v) with entries in
[-1, 0). That bounded block is the entire reason this method is better
conditioned than the explicit one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from i2pd.qp import (
    QP,
    FloatArray,
    duality_gap,
    residual_equality,
    residual_inequality,
    residual_stationarity,
)
from i2pd.retraction import b_mu, db_mu
from i2pd.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class SolverConfig:
    """Hyperparameters for the implicit IPM solver.

    Defaults that are not fixed by the paper's equations are noted; sigma is
    the centering parameter from Eq. 6 (sigma in (0, 1]).
    """

    sigma: float = 0.8  # centering parameter, Eq. 6. sigma in (0,1].
    max_iters: int = 100  # Newton iteration cap.
    tol_stat: float = 1e-4  # ||r_x||           stopping tolerance.
    tol_prim: float = 1e-4  # max(||r_i||,||r_e||) stopping tolerance.
    tol_gap: float = 1e-4  # duality gap        stopping tolerance.
    ls_armijo: float = 1e-4  # Armijo sufficient-decrease coefficient.
    ls_backtrack: float = 0.9  # line-search step shrink factor in (0,1).
    ls_alpha_min: float = 1e-12  # smallest step before declaring failure.

    def __post_init__(self) -> None:
        if not (0.0 < self.sigma <= 1.0):
            raise ValueError(f"sigma must be in (0, 1], got {self.sigma}")
        if self.max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        if not (0.0 < self.ls_backtrack < 1.0):
            raise ValueError("ls_backtrack must be in (0, 1)")


@dataclass
class SolverResult:
    """Outcome of an implicit IPM solve.

    Final primal/dual variables plus per-iteration scalar logs (gap and the
    three residual norms) for convergence inspection.
    """

    x: FloatArray
    lam: FloatArray
    gamma: FloatArray
    s: FloatArray
    v: FloatArray
    converged: bool
    iterations: int
    gap_history: list[float] = field(default_factory=list)
    stat_history: list[float] = field(default_factory=list)
    prim_history: list[float] = field(default_factory=list)


def _starting_point(qp: QP) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Construct an initial (x, lambda, gamma, s, v).

    We solve the *relaxed* KKT system in which the complementarity block is
    replaced by the identity (i.e. lambda = s formally), then shift lambda and
    s to be strictly positive so the barrier is well defined. v is recovered
    by inverting the retraction: from lambda = b_mu(v) and the softplus form
    b_mu(v) = (v + sqrt(v^2 + 4 mu)) / 2 we get v = lambda - mu/lambda.

    This is a standard interior-point warm start. It is not dictated by a
    single paper equation; any strictly-feasible-in-(lambda,s) start works.
    """
    n, m, p = qp.n, qp.m, qp.p

    # Relaxed KKT:  [ Q   -Aᵀ  -Cᵀ ] [x]   [-q]
    #               [ A    I    0  ] [λ] = [ b]
    #               [ C    0    0  ] [y]   [ d]
    top = np.hstack([qp.Q, -qp.A.T, -qp.C.T])
    mid = np.hstack([qp.A, np.eye(m), np.zeros((m, p))])
    bot = np.hstack([qp.C, np.zeros((p, m)), np.zeros((p, p))])
    kkt0 = np.vstack([top, mid, bot])
    rhs0 = np.concatenate([-qp.q, qp.b, qp.d])

    sol = scipy.linalg.solve(kkt0, rhs0)
    x = sol[:n]
    lam = sol[n : n + m]
    gamma = sol[n + m : n + m + p]
    s = -lam

    # Shift lambda, s strictly positive (eps0 = 1 is a common choice).
    eps0 = 1.0
    a_lam = float(np.min(lam)) if m > 0 else eps0
    a_s = float(np.min(s)) if m > 0 else eps0
    if eps0 > a_lam:
        lam = lam + (eps0 - a_lam) * np.ones(m)
    if eps0 > a_s:
        s = s + (eps0 - a_s) * np.ones(m)

    gap = float(lam @ s) if m > 0 else 0.0
    mu = gap / m if m > 0 else 0.0
    # Invert the softplus retraction:  v = lambda - mu / lambda.
    v = lam - mu / lam if m > 0 else np.zeros(0)
    return x, lam, gamma, s, v


def _residuals(
    qp: QP,
    x: FloatArray,
    lam: FloatArray,
    gamma: FloatArray,
    s: FloatArray,
    v: FloatArray,
    beta: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """The five residual blocks of Eq. 9, evaluated at barrier level beta.

    r_x       = Qx + q - Aᵀλ - Cᵀy        (stationarity)
    r_i       = Ax - b - s                 (inequality primal feasibility)
    r_e       = Cx - d                     (equality primal feasibility)
    r_lam_mu  = λ - b_mu(v)                (implicit complementarity, Eq. 9)
    r_s_mu    = s - b_mu(-v)               (implicit complementarity, Eq. 9)
    """
    r_x = residual_stationarity(qp, x, lam, gamma)
    r_i = residual_inequality(qp, x, s)
    r_e = residual_equality(qp, x)
    r_lam_mu = lam - b_mu(v, beta)
    r_s_mu = s - b_mu(-v, beta)
    return r_x, r_i, r_e, r_lam_mu, r_s_mu


def solve_implicit(qp: QP, config: SolverConfig | None = None) -> SolverResult:
    """Solve the QP with the implicit interior-point primal-dual method."""
    cfg = config if config is not None else SolverConfig()
    n, m, p = qp.n, qp.m, qp.p

    x, lam, gamma, s, v = _starting_point(qp)
    converged = False
    _it = 0
    gap_history: list[float] = []
    stat_history: list[float] = []
    prim_history: list[float] = []

    for _it in range(1, cfg.max_iters + 1):
        # --- barrier level: mu = sigma * eta / m, eta = λᵀs (Eq. 6) -------
        gap = duality_gap(lam, s) if m > 0 else 0.0
        mu = gap / m if m > 0 else 0.0
        beta = cfg.sigma * mu

        # --- residuals and convergence check ----------------------------
        r_x, r_i, r_e, r_lam_mu, r_s_mu = _residuals(qp, x, lam, gamma, s, v, beta)
        n_stat = float(np.linalg.norm(r_x))
        n_prim = max(float(np.linalg.norm(r_i)), float(np.linalg.norm(r_e)))
        gap_history.append(gap)
        stat_history.append(n_stat)
        prim_history.append(n_prim)

        _log.info(
            "iteration",
            it=_it,
            gap=gap,
            mu=mu,
            r_stat=n_stat,
            r_prim=n_prim,
        )

        if n_stat <= cfg.tol_stat and n_prim <= cfg.tol_prim and gap <= cfg.tol_gap:
            converged = True
            break

        # --- assemble reduced J(v) and rhs (Eq. 13) ---------------------
        # Diagonal retraction-derivative blocks.
        db_plus = db_mu(v, beta)  # db_mu(v),   used in back-substitution
        db_minus = db_mu(-v, beta)  # db_mu(-v),  the bounded [-1,0) block
        w = db_plus + db_minus  # = 1 for softplus (Def. 1, 14b)

        B_minus = np.diag(db_minus)  # B_mu(-v)
        W = np.diag(w)  # W = B_mu(v) + B_mu(-v)

        # J(v) = [ Q - AᵀA   -AᵀW   -Cᵀ ]
        #        [   -A      -B(-v)   0  ]
        #        [   -C        0      0  ]
        j_top = np.hstack([qp.Q - qp.A.T @ qp.A, -qp.A.T @ W, -qp.C.T])
        j_mid = np.hstack([-qp.A, -B_minus, np.zeros((m, p))])
        j_bot = np.hstack([-qp.C, np.zeros((p, m)), np.zeros((p, p))])
        jac = np.vstack([j_top, j_mid, j_bot])

        # rhs = [ -r_x + Aᵀ(r_i - r_lam_mu + r_s_mu) ]
        #       [          r_i + r_s_mu               ]
        #       [          r_e                        ]
        rhs = np.concatenate(
            [
                -r_x + qp.A.T @ (r_i - r_lam_mu + r_s_mu),
                r_i + r_s_mu,
                r_e,
            ]
        )

        # --- Newton solve for the compact unknowns [Δx; Δv; Δy] ----------
        delta = scipy.linalg.solve(jac, rhs)
        dx = delta[:n]
        dv = delta[n : n + m]
        dgamma = delta[n + m : n + m + p]

        # --- recover Δs, Δλ by back-substitution -------------------------
        #   Δs = A Δx + r_i           (from row 2 of the full system)
        #   Δλ = db_mu(v) ⊙ Δv - r_lam_mu   (from row 4 of the full system)
        ds = qp.A @ dx + r_i
        dlam = db_plus * dv - r_lam_mu

        # --- fraction-to-boundary + Armijo line search -------------------
        alpha = _line_search(
            qp,
            cfg,
            beta,
            x,
            lam,
            gamma,
            s,
            v,
            dx,
            dlam,
            dgamma,
            ds,
            dv,
            r_x,
            r_i,
            r_e,
            r_lam_mu,
            r_s_mu,
        )
        if alpha is None:
            _log.warning("line_search_failed", it=_it)
            break  # line search failed; return best-so-far, converged=False

        # --- take the step ----------------------------------------------
        x = x + alpha * dx
        lam = lam + alpha * dlam
        gamma = gamma + alpha * dgamma
        s = s + alpha * ds
        v = v + alpha * dv

    _log.info(
        "solve_finished",
        converged=converged,
        iterations=_it,
        final_gap=gap_history[-1] if gap_history else float("nan"),
    )
    return SolverResult(
        x=x,
        lam=lam,
        gamma=gamma,
        s=s,
        v=v,
        converged=converged,
        iterations=_it,
        gap_history=gap_history,
        stat_history=stat_history,
        prim_history=prim_history,
    )


def _merit(
    qp: QP,
    x: FloatArray,
    lam: FloatArray,
    gamma: FloatArray,
    s: FloatArray,
    v: FloatArray,
    beta: float,
) -> float:
    """Merit phi = 1/2 ||r||^2 over all five residual blocks.

    A standard line-search merit: the Newton direction decreases ||r||^2, so
    we accept a step that achieves Armijo sufficient decrease on phi.
    """
    r_x, r_i, r_e, r_lam_mu, r_s_mu = _residuals(qp, x, lam, gamma, s, v, beta)
    sq = (
        float(r_x @ r_x)
        + float(r_i @ r_i)
        + float(r_e @ r_e)
        + float(r_lam_mu @ r_lam_mu)
        + float(r_s_mu @ r_s_mu)
    )
    return 0.5 * sq


def _line_search(
    qp: QP,
    cfg: SolverConfig,
    beta: float,
    x: FloatArray,
    lam: FloatArray,
    gamma: FloatArray,
    s: FloatArray,
    v: FloatArray,
    dx: FloatArray,
    dlam: FloatArray,
    dgamma: FloatArray,
    ds: FloatArray,
    dv: FloatArray,
    r_x: FloatArray,
    r_i: FloatArray,
    r_e: FloatArray,
    r_lam_mu: FloatArray,
    r_s_mu: FloatArray,
) -> float | None:
    """Backtracking line search.

    Accept the largest alpha = ls_backtrack^k such that:
      (1) lambda and s stay >= 0 at the trial point (fraction-to-boundary), and
      (2) phi(trial) <= (1 - ls_armijo * alpha) * phi(current)  (Armijo).
    Returns the accepted alpha, or None if alpha falls below ls_alpha_min.
    """
    phi0 = 0.5 * (
        float(r_x @ r_x)
        + float(r_i @ r_i)
        + float(r_e @ r_e)
        + float(r_lam_mu @ r_lam_mu)
        + float(r_s_mu @ r_s_mu)
    )
    alpha = 1.0
    while alpha >= cfg.ls_alpha_min:
        lam_t = lam + alpha * dlam
        s_t = s + alpha * ds
        nonneg = bool(np.all(lam_t >= 0.0) and np.all(s_t >= 0.0))
        if nonneg:
            x_t = x + alpha * dx
            gamma_t = gamma + alpha * dgamma
            v_t = v + alpha * dv
            phi_t = _merit(qp, x_t, lam_t, gamma_t, s_t, v_t, beta)
            if phi_t <= (1.0 - cfg.ls_armijo * alpha) * phi0:
                return alpha
        alpha *= cfg.ls_backtrack
    return None
