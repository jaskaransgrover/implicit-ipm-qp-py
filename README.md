# implicit-ipm-qp

Python reproduction of **Implicit Primal-Dual Interior-Point Methods for Quadratic Programming**
(Arrizabalaga & Manchester, arXiv:2604.00364).

Reference Julia implementation: https://github.com/jonarriza96/i2pd.jl

## Development

```bash
uv sync --extra dev
uv run ruff check
uv run mypy
uv run pytest
```