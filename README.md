# Age-gradient CFR change test

Reproducible code and manuscript for a retrospective test of one change in
the age gradient of delay-adjusted case-fatality intensity. Daily cases are
convolved with age-specific case-to-death delays, and the age allocation of
daily deaths is analyzed conditionally on the daily death total.

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

The default backend is compiled C. On Windows the build searches for MinGW
GCC or LLVM Clang; every method also has a numerically equivalent pure-Python
backend.

## Direct method use

```python
import numpy as np
from methods import run_test

rng = np.random.default_rng(20260809)
result = run_test(
    "dags", exposure, deaths, age_scores, rng,
    backend="c", samples=5000,
)
print(result.statistic, result.p_value, result.estimated_change)
```

Stable method names are `dags`, `sup_lm`, `cvm`, and `no_delay_dags`. DAGS is
the delay-adjusted age-gradient scan. Use `backend="python"` for the reference implementation.
`bootstrap_change_intervals` repeats the complete change-day search in each
conditional-bootstrap sample.

## Reproduction

```bash
# Fast pipeline check: 100 replications per cell
python simulation.py --mode demo --backend c
python analysis.py --mode demo

# Manuscript results: 1000 replications per cell
python simulation.py --mode full --backend c
python analysis.py --mode full

# Primary SIVEP-Gripe application (requires the documented local Parquet data)
python realdata.py --bootstrap-samples 2000

```

Simulation jobs are parallel and restartable. Generated raw results, tables,
and figures are written to local output directories that are intentionally
excluded from this core-code repository.

## Repository layout

- `methods.py`: public test API and equivalent C/Python algorithms
- `dgp.py`: all data-generating mechanisms
- `simulation.py`: restartable demo/full Monte Carlo driver
- `analysis.py`: tables and publication PDF figures
- `realdata.py`: primary real-data application
- `config.py`: all fixed settings
- `csrc/cfr_core.c`: compiled backend source

Random seeds and scenario grids are fixed in `config.py`. Surveillance data,
generated results, compiler artifacts, the manuscript, tests, and local
archives are retained locally and excluded from this core-code repository by
`.gitignore`.

## License

Code is released under the MIT License. The manuscript and figures remain
subject to the authors' chosen publication license.
