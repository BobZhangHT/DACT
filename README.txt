Reproducing the DACT manuscript

These instructions implement the Biometrical Journal code/data checklist
dated 17 October 2025. The repository provides source code and compact saved
summaries; the journal's Data and Software ZIP additionally contains the frozen
patient-level input snapshots and full supporting files. GitHub alone does not
replace that submission artifact.

Installation and execution order

Use Python 3.11 or later, create and activate a virtual environment as described
in README.md, then run python -m pip install -r requirements.txt. For the
author's pinned dependency versions, use requirements-reproduction.txt with
Python 3.12.10. The recorded author environment is in environment.txt. The
programs do not install dependencies or modify global environment variables.

Run commands from the project root. The master resolves its inputs relative to
its own location; no source edits or machine-specific data paths are required.

python reproduce.py --mode environment
python reproduce.py --mode smoke --backend c
python reproduce.py --mode cached

environment saves current OS, Python, library, compiler and revision details.
smoke reruns three historical full-study replications using their original
seed identities and compares saved statistics, p-values, dates and decisions.
cached verifies reference hashes and renders all manuscript displays, without
refitting the models or reading individual patient records.

Generated files go under results_conditional_delay/reproduction/<mode>/.
Each mode records its environment and results in run.json. --output-dir
selects the destination for that record and for cached displays. Full/demo study
scripts keep their established results_conditional_delay and paper output
locations, regardless of this option. Changing it does not create an isolated
or fresh simulation run. The source reference files are read-only inputs.

Complete study and selected checks

python reproduce.py --mode full --workers 4 --dry-run
python reproduce.py --mode full --workers 4
python reproduce.py --mode spot --experiment testing --cell 4 --replication 0
python reproduce.py --mode demo --workers 4

The complete workflow runs simulations, aggregates them, analyzes the application,
compares delay supports and runs the timing benchmark, in that order. Prepare
the two input snapshots before executing application steps; see DATA.md.
--dry-run lists full/demo commands without executing those study steps.
--steps simulations analysis selects only the simulation study. Existing raw
records are resumed; remove or relocate an existing result file yourself only
when you intend a completely fresh run. A full run uses the paper's 1000
replications per design, 1000 simulation-calibration paths, 2000 interval
bootstrap draws and 5000 application-calibration paths.

Demo mode uses the same designs with 100 replications per cell. It runs only
simulation and aggregation steps by default. It is a reduced study, not an
instant example; smoke is the quick numerical check. Full-study runtime depends
on CPU and worker count. The saved summaries support immediate display checks.

spot accepts any valid cell and replication without constructing the full
task grid. The supplied historical references cover testing cells 0, 4 and 144
at replication 0. Other selections are saved with saved_reference_match: null
unless a historical reference exists. Timings are excluded from numerical
comparisons, and floating-point comparisons use absolute/relative tolerance
1e-9. Arbitrary spot checks are not evidence that the entire study was rerun.

Random numbers and intermediate results

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

Display and numerical-result map

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

Compiled code and testing

The default C backend is built from csrc/cfr_core.c. A compiler must be on PATH,
or CC may name a compiler executable. Build the library and print its path with:

python -c "from methods import build_c_backend; print(build_c_backend(force=True))"

The actual argument list is available from methods._compile_command.

Windows uses -O3 -std=c11 -shared -static-libgcc; Linux uses
-O3 -std=c11 -shared -fPIC plus -lm; macOS uses
-O3 -std=c11 -dynamiclib -fPIC plus -lm. Outputs are respectively a
source-hashed .dll, .so or .dylib under csrc/. Only source code is tracked in Git.
Native binaries supplied with a journal supplement must be labeled by platform
and accompanied by this source and the recorded compiler information.

The public numerical API and simulation driver also support the Python backend.
Run python reproduce.py --mode smoke --backend python for its saved-result
check. Run python -m pytest -q for the full test suite. The supplied CI workflow
configures a Linux check; adding a workflow is not a claim that the remote run
has already passed. Local verification is recorded separately.

File structure

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


AUTHOR VALIDATION ENVIRONMENT - 2026-10-18

{
  "platform": "Windows-11-10.0.26200-SP0",
  "machine": "AMD64",
  "python": "3.12.10",
  "packages": {
    "numpy": "2.4.6",
    "scipy": "1.16.3",
    "pandas": "2.3.3",
    "matplotlib": "3.11.0",
    "pyarrow": "25.0.0",
    "tqdm": "4.68.3",
    "pytest": "9.1.1"
  },
  "compiler": {
    "name": "gcc.EXE",
    "version": "gcc.EXE (MinGW-W64 x86_64-ucrt-mcf-seh, built by Brecht Sanders, r1) 16.1.0"
  },
  "revision": "3a6615db396245bd6611b300a74ac4749cee8324",
  "working_tree_modified": true,
  "source_sha256": {
    "config.py": "732da6aa37b541e38eaea36f01e2c701ba4d7acd96be1bdd51d7551139eb74bd",
    "dgp.py": "b3764820cf5767238bfdfe2fa6301c24bafa6dd7dac908e2da1cad64b812227e",
    "methods.py": "9177a1eaa0afa0044217f85593ed6b4406b4ea1132e146a3ea4d9edd2c87c217",
    "simulation.py": "edc093ebc83b70432f51294a9463d52b93b26b94b31d5c1093d3f1049035a4db",
    "analysis.py": "afa4d07ffe80f8150fb853ca2947c9a19a88b8112c600d2ec880d836c12936ce",
    "realdata.py": "ff381587f7509570fa4a4b3b83af258de09057ffac34fbc32323c30c45161595",
    "reproduce.py": "f41435c223c64fdef31bdbe30bf43ecfa2e3e2b75703e089d2b765150f99bbb5",
    "csrc/cfr_core.c": "9c0120b1269f918b17767a660053e2e1997b5fece5b16527da14b2a759135449",
    "scripts/realdata_sensitivity.py": "10e982e81562fe9d7fcb40129ad019e059dace827ddc946591a1d5eefbf8626d",
    "scripts/benchmark.py": "b4c1fbbce0f05fb1db0271017ccc9286dd29a325d34cfc45e8fdfd9982600239"
  }
}

INSTALLED PINNED DEPENDENCY CLOSURE
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
fonttools==4.63.0
iniconfig==2.3.0
kiwisolver==1.5.0
matplotlib==3.11.0
numpy==2.4.6
packaging==26.2
pandas==2.3.3
pillow==12.2.0
pluggy==1.6.0
pyarrow==25.0.0
Pygments==2.20.0
pyparsing==3.3.2
pytest==9.1.1
python-dateutil==2.9.0.post0
pytz==2026.3.post1
scipy==1.16.3
six==1.17.0
tqdm==4.68.3
tzdata==2026.3

The 2026-10-18 checks are not a full original-input rerun on an independent machine.


Data sources, variables and access

Original surveillance records

Source and rights holder: Brazilian Ministry of Health / DATASUS,
SIVEP-Gripe severe acute respiratory illness surveillance.

- Dataset catalogue: https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026
- Analysis dictionary: https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/pdfs/Dicionario_de_Dados_SRAG_Hospitalizado_19.09.2022.pdf
- Local input fingerprints and catalogue identifiers: data_sources.json.

The MIT license covers this repository's software, not third-party surveillance
records. The official portal currently displays a Creative Commons
Attribution-NoDerivatives 3.0 site notice; source-specific rights and reuse terms
remain governed by the Ministry. No broader data license is asserted here.
The repository does not redistribute individual patient records. The local
journal reproduction archive contains the two frozen input snapshots; a public
archive DOI or download link has not been assigned by this repository.

Place INFLUD21-23-03-2026.parquet and INFLUD22-23-03-2026.parquet under
data/sivep/. They contain 1745672 and 560577 rows before selection.
Original download timestamps and conversion logs were not retained. The
filename dates identify the local snapshots and are not independent evidence
of retrieval dates. SHA-256 values identify the exact bytes used locally.
Verify those fingerprints before claiming exact manuscript reproduction.

Variables used by realdata.py

| Field | Meaning and use |
|---|---|
| NU_NOTIFIC | Internal notification identifier; last entry retained per identifier |
| DT_NOTIFIC | Notification-form date; orders repeated entries |
| DT_SIN_PRI | Symptom-onset date; case calendar and eligibility |
| DT_EVOLUCA | Discharge/death date; fatal records supply death dates |
| NU_IDADE_N | Recorded numerical age; retained range 0-110 |
| TP_IDADE | Age unit; 3 denotes age recorded in years |
| CLASSI_FIN | Final classification; 5 selects COVID-19 |
| CRITERIO | Confirmation criterion; 1 selects laboratory confirmation |
| HOSPITAL | Hospitalization indicator; 1 selects hospitalized records |
| EVOLUCAO | Outcome; 2 selects deaths for the dated death series |

The six groups are 0-39, 40-49, 50-59, 60-69, 70-79 and 80+ years.
Representative age scores are (-3, -1.5, -0.5, 0.5, 1.5, 2.5) in decades.
All eligible outcomes contribute to case history; dated fatal outcomes form
the death series. The 2022 death window uses 120 days of prior case history.
The baseline delay estimator uses fatal records with onset from January through
September 2021 and retains nonnegative onset-to-death delays up to 90 days.
See realdata._read, _delay and _daily_arrays for the exact operations.

Generated simulation and cached data

Simulation inputs are generated entirely by dgp.generate_dataset. N0-N3 are
constant-age-effect mechanisms; A1-A3 represent flattening, steepening and
sparse flattening. Each SimulationData object stores daily cases, aligned
exposures, deaths, age scores, true and analyzed delays, the true coefficient
path and the true split. It contains no real patients.

reference/full_testing.csv is keyed by scenario, method, days, death scale,
effect magnitude and change fraction. It reports rejection, replication count,
fit failure rate, runtime and median location error. full_estimation.csv is
keyed by scenario, design and parameter, and reports bias, empirical SD, mean
bootstrap SE, coverage, width and successful-replication count.
full_sensitivity.csv is keyed by scenario, perturbation axis and value.

application_arrays.npz stores aggregate daily exposures/deaths and delay masses,
with the score and information-time arrays used for Figure 3. Dates are safe
Unicode arrays. application_results.json stores test values, split, segment
effects and model-based percentile intervals. The sensitivity CSV and JSON
store cutoff-specific summaries and delay-tail diagnostics for Figure 4.
manifest.json verifies every cached file. spotchecks.json retains selected
historical replication outputs for direct numerical comparison.

These aggregates permit display checks without accessing individual records;
they do not substitute for the original records when reproducing cohort
selection, preprocessing or fitted delay distributions.
