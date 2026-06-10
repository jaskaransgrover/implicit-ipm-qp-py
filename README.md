<div align="center">

# `implicit-ipm-qp-py`

### A Python reproduction of *Implicit Interior-Point Methods for Quadratic Programming*

[![arXiv](https://img.shields.io/badge/arXiv-2604.00364-B31B1B.svg)](https://arxiv.org/abs/2604.00364)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Checks](https://img.shields.io/badge/ruff%20%7C%20mypy%20%7C%20pytest-passing-success.svg)](#development)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

</div>

A clean, well-tested **NumPy/SciPy** reproduction of the *implicit* interior-point
primal-dual method for convex quadratic programs, from the paper by
Arrizabalaga & Manchester ([arXiv:2604.00364](https://arxiv.org/abs/2604.00364)).

The solver tackles

$$\min_{x}\ \tfrac{1}{2}x^\top Q x + q^\top x \quad \text{s.t.}\quad Ax \ge b,\ \ Cx = d,\qquad Q \succeq 0.$$

> [!NOTE]
> This is an independent, from-first-principles Python reimplementation built for
> education purposes. The original reference implementation from the paper's authors is in Julia:
> [jonarriza96/i2pd.jl](https://github.com/jonarriza96/i2pd.jl).

## Why "implicit"?

Classical interior-point methods relax complementarity to $\lambda_i s_i = \mu$ and,
after condensing, must form the diagonal block $-\Lambda^{-1}S$ whose entries
$-s_i/\lambda_i$ blow up toward $0$ or $\infty$ as the barrier $\mu \to 0$. That is
the source of the notorious ill-conditioning near the solution.

The authors' proposed implicit method instead parameterizes the multipliers and slacks through a
**retraction map** $b_\mu$:

$\lambda = b_\mu(v),\qquad s = b_\mu(-v),\qquad v \in \mathbb{R}^m,$

chosen so that $b_\mu(v)\,b_\mu(-v) = \mu$ and $\partial b_\mu(v) + \partial b_\mu(-v) = 1$ hold
by construction. After condensing, the analogous block becomes $-B_\mu(-v)$ with
entries confined to $[-1, 0)$ — a bounded spectrum, no matter how small $\mu$
gets. That single change leads to a Newton system that stays
well-conditioned all the way to convergence.

This repo uses the **softplus** retraction

$$b_\mu(v) = \tfrac{1}{2}\bigl(v + \sqrt{v^2 + 4\mu}\bigr).$$

## Installation

> [!IMPORTANT]
> Requires Python **3.12**. We recommend a [`uv`](https://github.com/astral-sh/uv)-managed
> environment.

From source:

```bash
git clone https://github.com/jaskaransgrover/implicit-ipm-qp-py
cd implicit-ipm-qp-py
uv sync --extra dev
```

## Hello world

Solve $\min \tfrac12(x_1^2 + x_2^2)$ subject to $x_1 + x_2 \ge 2$. The constraint is
active, so the optimum sits at $(1, 1)$.

```python
import numpy as np

from implicit_ipm_qp.qp import QP
from implicit_ipm_qp.solvers.implicit import solve_implicit

qp = QP(
    Q=np.eye(2),
    q=np.zeros(2),
    A=np.array([[1.0, 1.0]]),   # x1 + x2 ...
    b=np.array([2.0]),          # ... >= 2
    C=np.zeros((0, 2)),         # no equality constraints
    d=np.zeros(0),
)

result = solve_implicit(qp)

print("converged:", result.converged)
print("x*       :", result.x)          # ~ [1. 1.]
print("lambda*  :", result.lam)        # active-constraint multiplier > 0
print("iters    :", result.iterations)
```

Tune the solve through `SolverConfig`:

```python
from implicit_ipm_qp.solvers.implicit import SolverConfig

cfg = SolverConfig(sigma=0.8, tol_gap=1e-8, max_iters=200)
result = solve_implicit(qp, cfg)
```

## What's inside

| Module | Contents |
| --- | --- |
| `implicit_ipm_qp.qp` | The `QP` problem type and residual functions (stationarity, primal feasibility, complementarity, duality gap). |
| `implicit_ipm_qp.retraction` | The softplus retraction `b_mu` and its derivative `db_mu`, with property tests for the defining identities. |
| `implicit_ipm_qp.scaling` | Ruiz equilibration for problem preconditioning. |
| `implicit_ipm_qp.solvers.implicit` | The implicit interior-point solver: reduced $J(v)$ Newton system, back-substitution, and a fraction-to-boundary + Armijo line search. |

## Development

The project is checked with `ruff` (lint + format), `mypy --strict`, and `pytest`:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest -v
```

All four run in CI on every push.

## Roadmap

- [x] Core QP data model, residuals, Ruiz equilibration
- [x] Softplus retraction map with property tests
- [x] Implicit interior-point solver
- [ ] Explicit baseline solver (for the conditioning comparison)
- [ ] Toy-QP reproduction (paper Fig. 3)
- [ ] Maros–Mészáros benchmark suite
- [ ] Factorization-free MINRES path (paper Fig. 5)
- [ ] `float32` vs `float64` precision study (paper Fig. 6)

## Citation

If you use this reproduction, please cite the original paper:

```bibtex
@article{arrizabalaga2026implicit,
  title   = {Implicit Interior-Point Methods for Quadratic Programming},
  author  = {Arrizabalaga, Jon and Manchester, Zachary},
  journal = {arXiv preprint arXiv:2604.00364},
  year    = {2026}
}
```

## License

MIT