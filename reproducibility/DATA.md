# Data sources, variables and access

## Original surveillance records

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

## Variables used by realdata.py

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

## Generated simulation and cached data

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
