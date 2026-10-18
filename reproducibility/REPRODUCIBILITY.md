# Reproducing the DACT manuscript

These instructions implement the Biometrical Journal code/data checklist
dated 17 October 2025. The repository provides source code and compact saved
summaries; the journal's Data and Software ZIP additionally contains the frozen
patient-level input snapshots and full supporting files. GitHub alone does not
replace that submission artifact.

## Installation and execution order

Use Python 3.11 or later, create and activate a virtual environment as described
in README.md, then run `python -m pip install -r requirements.txt`. This file
pins the author's dependency versions, tested with Python 3.12.10.
The recorded author environment is in environment.txt. The
programs do not install dependencies or modify global environment variables.

Run commands from the project root. The master resolves its inputs relative to
its own location; no source edits or machine-specific data paths are required.

```bash
python reproduce.py --mode environment
python reproduce.py --mode smoke --backend c
python reproduce.py --mode cached
```

`environment` saves current OS, Python, library, compiler and revision details.
`smoke` reruns three historical full-study replications using their original
seed identities and compares saved statistics, p-values, dates and decisions.
`cached` verifies reference hashes and renders all manuscript displays, without
refitting the models or reading individual patient records.

Generated files go under `results_conditional_delay/reproduction/<mode>/`.
Each mode records its environment and results in `run.json`. `--output-dir`
selects the destination for that record and for cached displays. Full/demo study
scripts keep their established results_conditional_delay and paper output
locations, regardless of this option. Changing it does not create an isolated
or fresh simulation run. The source reference files are read-only inputs.

## Complete study and selected checks

```bash
python reproduce.py --mode full --workers 4 --dry-run
python reproduce.py --mode full --workers 4
python reproduce.py --mode spot --experiment testing --cell 4 --replication 0
python reproduce.py --mode demo --workers 4
```

The complete workflow runs simulations, aggregates them, analyzes the application,
compares delay supports and runs the timing benchmark, in that order. Prepare
the two input snapshots before executing application steps; see DATA.md.
`--dry-run` lists full/demo commands without executing those study steps.
`--steps simulations analysis` selects only the simulation study. Existing raw
records are resumed; remove or relocate an existing result file yourself only
when you intend a completely fresh run. A full run uses the paper's 1000
replications per design, 1000 simulation-calibration paths, 2000 interval
bootstrap draws and 5000 application-calibration paths.

Demo mode uses the same designs with 100 replications per cell. It runs only
simulation and aggregation steps by default. It is a reduced study, not an
instant example; `smoke` is the quick numerical check. Full-study runtime depends
on CPU and worker count. The saved summaries support immediate display checks.

`spot` accepts any valid cell and replication without constructing the full
task grid. The supplied historical references cover testing cells 0, 4 and 144
at replication 0. Other selections are saved with `saved_reference_match: null`
unless a historical reference exists. Timings are excluded from numerical
comparisons, and floating-point comparisons use absolute/relative tolerance
1e-9. Arbitrary spot checks are not evidence that the entire study was rerun.

## Random numbers and intermediate results

The simulation master seed is 20260809. Each data-generation and method stream
uses a NumPy SeedSequence combining that seed, the CRC32 of its stream label,
the design-cell index and the replication index. Worker order does not change
these identities. Application seeds are 20260809, 20260810 and 20260811 for
Gaussian calibration, null bootstrap and interval bootstrap, respectively.
The support comparison uses 20260809 plus the cutoff index. The benchmark's
data and method streams are documented in scripts/benchmark.py.

Every full/demo replication is saved immediately in a named JSONL result file.
Aggregated CSV files retain scenario/method/design keys and counts. Historical
summaries are kept in reference/ with SHA-256 checksums. Date arrays in the
application cache use Unicode strings and require no pickle loading. These
files contain aggregate series and summaries, not individual patient records.

Cached rendering also writes testing_monte_carlo_se.csv, using binomial
Monte Carlo standard errors sqrt(p*(1-p)/n). This quantifies finite-replication
variability; it does not establish stability under independent reruns. The
checklist's independent-seed and coauthor-machine confirmations remain author
validation steps; see CHECKLIST.md.

## Display and numerical-result map

| Manuscript item | Function or script | Saved input | Cached output |
|---|---|---|---|
| Table 1: scenario definitions | reproduce.scenario_table | Fixed study design | tables/table_1_scenarios.tex |
| Table 2: testing/localization | analysis.testing_table | full_testing.csv | tables/table_2_testing.tex |
| Table 3: parameter estimation | analysis.estimation_table | full_estimation.csv | tables/table_3_estimation.tex |
| Figure 1: power | analysis.power_figure | full_testing.csv | figures/figure_power_surface.pdf |
| Figure 2: simulation sensitivity | analysis.sensitivity_figures | full_sensitivity.csv | figures/figure_sensitivity.pdf |
| Figure 3: application | realdata._plot | application_results.json; application_arrays.npz | figures/figure_application.pdf |
| Figure 4: delay supports | scripts.realdata_sensitivity._plot | application_sensitivity.csv; application_sensitivity_metadata.json | figures/figure_application_sensitivity.pdf |
| Runtime paragraph | scripts.benchmark.benchmark | Seeded synthetic datasets | results_conditional_delay/benchmark.json |

Simulation null rejection, power and localization paragraphs use full_testing.csv;
estimation paragraphs use full_estimation.csv; specification checks use
full_sensitivity.csv. Section 4's numerical estimates and intervals use
application_results.json, with sensitivity estimates in application_sensitivity.csv.
Cached displays preserve reported statistical values; they are not a new full
reproduction of data preprocessing and fitting. Benchmark times are measurements
of a particular environment and are not expected to repeat bit for bit.

## Compiled code and testing

The default C backend is built from csrc/cfr_core.c. A compiler must be on PATH,
or CC may name a compiler executable. Build the library and print its path with:

```bash
python -c "from methods import build_c_backend; print(build_c_backend(force=True))"
```

The actual argument list is available from methods._compile_command.

Windows uses `-O3 -std=c11 -shared -static-libgcc`; Linux uses
`-O3 -std=c11 -shared -fPIC` plus `-lm`; macOS uses
`-O3 -std=c11 -dynamiclib -fPIC` plus `-lm`. Outputs are respectively a
source-hashed .dll, .so or .dylib under csrc/. Only source code is tracked in Git.
Native binaries supplied with a journal supplement must be labeled by platform
and accompanied by this source and the recorded compiler information.

The public numerical API and simulation driver also support the Python backend.
Run `python reproduce.py --mode smoke --backend python` for its saved-result
check. Run `python -m pytest -q` for the full test suite. The supplied CI workflow
configures a Linux check; adding a workflow is not a claim that the remote run
has already passed. Local verification is recorded separately.

## File structure

- config.py: fixed configuration, validation and explicit directory setup.
- dgp.py: simulation generators and true parameter paths.
- methods.py: likelihood fitting, change tests, bootstrap and C interface.
- csrc/cfr_core.c: documented native numerical kernels.
- simulation.py: seeded grid definitions, task execution and JSONL records.
- analysis.py: summary aggregation and simulation displays.
- realdata.py: data selection, alignment, application fitting and Figure 3.
- scripts/realdata_sensitivity.py: delay-support comparison and Figure 4.
- scripts/benchmark.py: single-process runtime measurements.
- reproduce.py: documented master, environment capture, cached and spot modes.
- reproducibility/reference/: small frozen summaries and their checksum manifest.
- tests/: numerical, workflow, formatting-independent and cache checks.

No source file should be edited to switch scenarios or methods. CLI controls
and the existing design grids perform those iterations. Generated data,
libraries, tests' temporary files and full-study outputs are excluded from Git.
