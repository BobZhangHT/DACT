# A Delay-Adjusted Change-Point Test for the Age Gradient of Case-Fatality Intensity

This repository contains the core code for the delay-adjusted change test
(DACT). DACT tests for one retrospective change in the age gradient of relative
fatality intensity using daily age-stratified case and death counts. It aligns
cases and deaths by an age-specific delay convolution and conditions on each
day's total number of deaths.

## Requirements

- Python 3.11 or later
- A C compiler for the default backend: GCC, Clang, or MinGW-w64

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. On macOS or Linux, use
`source .venv/bin/activate`.

The C backend is compiled automatically on first use. Every algorithm also has
a pure-Python implementation, selected with `backend="python"`.

## Quick Start

```python
import numpy as np

from dgp import generate_dataset
from methods import run_test

rng = np.random.default_rng(20260809)
data = generate_dataset("A1", rng, days=180, change_magnitude=0.025)
result = run_test(
    "dact",
    data.exposure,
    data.deaths,
    data.age_scores,
    np.random.default_rng(20260810),
    backend="c",
    samples=5000,
)

print(result.statistic, result.p_value, result.estimated_change)
```

The public method names are `dact`, `sup_lm`, `cvm`, and `no_delay_dact`.
The last method deliberately uses same-day cases and is included as a
misspecified comparison. `bootstrap_change_intervals` repeats the complete
change-date search in every conditional-bootstrap sample.

## Reproduce the Simulation Study

Demo mode uses 100 replications per cell; full mode uses 1000. Apart from the
replication count, both modes use the same scenarios and method settings.

```bash
# Fast pipeline check
python simulation.py --mode demo --backend c
python analysis.py --mode demo

# Manuscript-scale study
python simulation.py --mode full --backend c --resume
python analysis.py --mode full
```

Simulation jobs are parallel and restartable. Use `--workers N` to set the
number of worker processes and `--resume` to continue an interrupted run.
Generated results are written under `results_conditional_delay/` and are not
tracked by Git.

## Brazilian SIVEP-Gripe Application

The surveillance microdata are public but too large to distribute in this
repository. Place the following local Parquet files under `data/sivep/`:

```text
data/sivep/INFLUD21-23-03-2026.parquet
data/sivep/INFLUD22-23-03-2026.parquet
```

They must contain the columns listed in `realdata.py`: `NU_NOTIFIC`,
`DT_NOTIFIC`, `DT_SIN_PRI`, `DT_EVOLUCA`, `NU_IDADE_N`, `TP_IDADE`,
`CLASSI_FIN`, `CRITERIO`, `HOSPITAL`, and `EVOLUCAO`.

Run the primary analysis and the prespecified delay-cutoff comparison with:

```bash
python realdata.py --bootstrap-samples 2000
python -m scripts.realdata_sensitivity --gaussian-samples 5000
```

The second command repeats the full-year analysis with maximum retained delays
of 60, 90, and 120 days. It does not vary the analysis window.

## Tests

```bash
python -m pytest -q
```

The test suite checks the data-generating mechanisms, public API, restartable
simulation driver, figure generation, and numerical agreement between the C
and Python backends.

## Repository Layout

- `methods.py`: DACT, benchmark tests, estimation, and the public API
- `dgp.py`: all simulation data-generating mechanisms
- `simulation.py`: parallel, restartable demo/full simulation driver
- `analysis.py`: aggregation, LaTeX tables, and publication figures
- `realdata.py`: primary SIVEP-Gripe analysis
- `config.py`: fixed model, method, and simulation settings
- `csrc/cfr_core.c`: native C implementation
- `scripts/realdata_sensitivity.py`: 60/90/120-day delay comparison
- `tests/`: numerical and pipeline tests

Only these source files, tests, and repository metadata are admitted by the
allowlist-style `.gitignore`. Local data, generated results, manuscript files,
reference papers, compiled libraries, caches, and archives remain untracked.

## License and Citation

The code is released under the MIT License. Citation metadata are provided in
`CITATION.cff`.
