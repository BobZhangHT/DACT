# A Delay-Adjusted Test for Changes in Age-Specific Fatality Patterns

This repository contains the core code for the delay-adjusted change test
(DACT). DACT tests for one retrospective change in the relative age effect in
fatality intensity using daily age-stratified case and death counts. It aligns
cases and deaths by an age-specific delay convolution and conditions on each
day's total number of deaths. The outputs combine a test, an estimated
transition date, segment effects and search-repeated model-based intervals.
See [CHANGELOG.md](CHANGELOG.md) for the current update.

## Biometrical Journal Reproduction

The journal-format instructions are in [README.txt](README.txt). They include
software versions, data documentation, execution order and a figure/table map.

```bash
python reproduce.py --mode smoke --backend c
python reproduce.py --mode cached
```

The first command verifies three saved simulation replications. The second
redraws Figures 1-4 and Tables 1-3 from compact, checksummed aggregate inputs;
it needs no patient-level data. Outputs and run metadata are written under
`results_conditional_delay/reproduction/`.

For the complete study after preparing the original input snapshots:

```bash
python reproduce.py --mode full --workers 4 --dry-run
python reproduce.py --mode full --workers 4
```

See [reproduction instructions](reproducibility/REPRODUCIBILITY.md),
[data documentation](reproducibility/DATA.md), and the
[checklist record](reproducibility/CHECKLIST.md). The GitHub source and small
aggregate references complement the separate journal Data and Software ZIP.

## Requirements

- Python 3.11 or later
- A C compiler for the default backend: GCC, Clang, or MinGW-w64

Run commands from the repository root. Create a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. On macOS or Linux, use
`source .venv/bin/activate`. Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` contains the fixed dependency versions tested with Python 3.12.10.

The C backend is compiled automatically on first use. The numerical API also
supports `backend="python"`. The real-data command-line scripts use the C backend.

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
    backend="python",
    samples=5000,
)

print(result.statistic, result.critical_value, result.p_value)
print(result.rejected, result.estimated_change)
```

Use `backend="c"` for native execution. The public method names are `dact`,
`sup_lm`, `cvm`, and `no_delay_dact`.
The last method deliberately uses same-day cases and is included as a
misspecified comparison. `bootstrap_change_intervals` repeats the complete
change-date search in every conditional-bootstrap sample. DACT uses Gaussian
calibration by default; `run_test("dact", ..., calibration="bootstrap")`
uses fitted-null multinomial calibration.

### Inputs and Interpretation

- Exposures and deaths have shape `(T, J)`; age scores have shape `(J,)`.
  Counts must be finite and nonnegative, and deaths must be integers. A positive
  death count requires positive exposure in that cell.
- `delay_adjusted_exposure` takes case counts and normalized age-specific delay
  masses of shape `(J, L)`, with column zero representing a zero-day delay.
- `estimated_change` is the **number of days in the first segment**, not a
  zero-based index. Use `dates[result.estimated_change - 1]` for its last day.
- The default search retains 15%-85% of cumulative fitted information, rather
  than a fixed fraction of calendar days. The critical value calibrates the
  maximum over this search.
- Gaussian and bootstrap rank cutoffs are consistent with add-one p-values.
- With age scores measured in decades, `exp(beta)` is the relative
  death-intensity multiplier per decade; `exp(beta_after - beta_before)` is
  its after/before ratio within the modeled population.

Interval outputs include estimates, standard errors, percentile intervals,
draws and bootstrap diagnostics. These intervals condition on the modeled
exposures and delay probabilities.

## Reproduce the Simulation Study

Demo mode uses 100 replications per cell; full mode uses 1000. Apart from the
replication count, both modes use the same scenarios and method settings.

```bash
# Demo: fewer replications, with the same design settings
python simulation.py --mode demo --backend c
python analysis.py --mode demo

# Manuscript-scale study
python simulation.py --mode full --backend c --resume
python analysis.py --mode full
python -m scripts.benchmark
```

Simulation jobs are parallel and restartable. Use `--workers N` to set the
number of worker processes and `--resume` to continue an interrupted run.
Both modes include many cells and search-repeated bootstraps; use Quick Start
for a single-test example. The driver accepts `--experiments testing`,
`--experiments intervals` or `--experiments sensitivity`, and `--backend python`
for simulations without a compiler.

Raw records and aggregated CSV summaries are generated under
`results_conditional_delay/`. Demo figures and tables go to its
`aggregated/demo_figures/` and `aggregated/demo_tables/` directories. Full-mode
figures and tables go to `paper/figures/` and `paper/tables/`. These directories
are created locally and remain untracked.

## Brazilian SIVEP-Gripe Application

The application uses hospitalized, laboratory-confirmed COVID-19 records from
Brazil's public [SIVEP-Gripe portal](https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026).
The death window covers 2022, with 120 days of earlier case history. Age-specific
onset-to-death delays are fitted from eligible fatal records with onset during
January-September 2021; the baseline retains delays of 0-90 days.

Microdata are not distributed in this repository. Place the following local
Parquet files under `data/sivep/`:

```text
data/sivep/INFLUD21-23-03-2026.parquet
data/sivep/INFLUD22-23-03-2026.parquet
```

They must contain the columns listed in `realdata.py`: `NU_NOTIFIC`,
`DT_NOTIFIC`, `DT_SIN_PRI`, `DT_EVOLUCA`, `NU_IDADE_N`, `TP_IDADE`,
`CLASSI_FIN`, `CRITERIO`, `HOSPITAL`, and `EVOLUCAO`.

Use the frozen snapshots above for manuscript reproduction.

Run the primary analysis and the prespecified delay-cutoff comparison with:

```bash
python realdata.py --bootstrap-samples 2000 --seed 20260809
python -m scripts.realdata_sensitivity --gaussian-samples 5000 --seed 20260809
```

The first command uses 5000 Gaussian paths and 2000 draws for each null
calibration and interval bootstrap. The second repeats the full-year analysis
with delay supports of 60, 90, and 120 days, keeping the case history and analysis
window fixed. Output metadata include SHA-256 hashes of the actual inputs.

Outputs under `results_conditional_delay/application/` include:

| File | Contents |
|---|---|
| `application_results.json` | Test results, selected date, segment effects, intervals and bootstrap diagnostics |
| `application_arrays.npz` | Daily series, delays, score process and information time |
| `application_sensitivity.csv` | Results at the three delay supports |
| `application_sensitivity_metadata.json` | Settings, delay diagnostics and actual input hashes |

Plots are written to `paper/figures/`; the support-comparison plot is also saved
alongside application results. The main application figure shows the absolute
score, the global model-based 5% threshold and the information-time search.
The selected date describes a surveillance transition; causal attribution
requires an appropriate patient-level study design.

## Tests

```bash
python -m pytest -q
```

The suite checks data-generating mechanisms, public API, simulation bookkeeping,
plots, input validation, structural zeros, rank calibration, bootstrap failure
handling, date ties, input fingerprints and C/Python numerical agreement.
C-backend tests need a compiler. No surveillance microdata are needed.

## Repository Layout

- `methods.py`: DACT, benchmark tests, estimation, and the public API
- `dgp.py`: all simulation data-generating mechanisms
- `simulation.py`: parallel, restartable demo/full simulation driver
- `analysis.py`: aggregation, LaTeX tables, and publication figures
- `realdata.py`: primary SIVEP-Gripe analysis
- `config.py`: fixed model, method, and simulation settings
- `csrc/cfr_core.c`: native C implementation
- `scripts/realdata_sensitivity.py`: 60/90/120-day delay comparison
- `scripts/benchmark.py`: reproducible single-process runtime benchmark
- `tests/`: numerical and pipeline tests

Only these source files, tests, and repository metadata are admitted by the
allowlist-style `.gitignore`, together with the documented compact references
under `reproducibility/reference/`. Local raw data, generated results, manuscript files,
reference papers, compiled libraries, caches, and archives remain untracked.

## License and Citation

The code is released under the MIT License. Citation metadata are provided in
`CITATION.cff`.
