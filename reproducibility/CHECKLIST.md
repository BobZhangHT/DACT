# Biometrical Journal code/data checklist: implementation record

Reference policy: Checklist for Code and Data Supplements, 2025-10-17,
https://onlinelibrary.wiley.com/pb-assets/assets/15214036/ag/Reproducible-Research-Author-Checklist-2025-1761054289830.pdf

Current author guide:
https://onlinelibrary.wiley.com/page/journal/15214036/homepage/forauthors.html

This record concerns the GitHub source revision. It is not a completed author
attestation for items that require a full rerun, independent machine or actual
submission upload.

| Checklist requirement | Repository implementation | Status |
|---|---|---|
| Master scripts; no repeated manual edits | reproduce.py; simulation.make_tasks/make_task | Implemented |
| Named per-replication intermediate results | raw/<mode>_<experiment>.jsonl; aggregated CSV | Implemented |
| Figure/table-to-code map | REPRODUCIBILITY.md and source docstrings | Implemented |
| Automatic display saving | Cached and full plotting/table functions | Implemented |
| Explicit seeds | SeedSequence task identities and documented application seeds | Implemented |
| Intermediate-results spot checks | Historical saved cells and reproduce.py smoke/spot | Verified locally for supplied references |
| README in TXT/HTML/PDF | Root README.txt | Included |
| Software and OS/compiler versions | requirements-reproduction.txt; environment.txt; per-run JSON | Included |
| Data provenance, variables and rights | DATA.md; data_sources.json; reference manifest | Included; original conversion logs unavailable |
| English names and comments; UTF-8; consistent style | Documented Python/C functions, Ruff and clang-format settings | Checked locally |
| Relative project paths | __file__-based paths; compiler from PATH or CC | Implemented |
| Definitions separate from execution | Explicit run/main entry points; no import-time directory creation | Implemented |
| Portable source and compilation instructions | Python backend; platform-aware C flags; documented native sources | Windows checked locally; remote platforms require CI confirmation |
| All article results reproduced from original inputs | Complete full workflow supplied | Full cross-machine rerun remains author validation |
| Independent-seed Monte Carlo stability | Per-cell binomial MCSE exported; fixed-seed comparisons supplied | Independent reruns remain author validation |
| Coauthor reproduces on own machine | Master and instructions supplied | Author action |
| Complete original case-study data and intermediate files | Separate local Data and Software package | Validate archive contents before journal upload; GitHub excludes raw patient data |
| Compiled source plus platform-labeled precompiled artifacts | Source, build instructions and a source-hashed native library in the local journal ZIP; manifest records its Windows/compiler environment | Windows artifact tested locally; other platforms build from source |
| Single ZIP uploaded as Data and Software | Local archive is separate from GitHub | Actual journal upload remains author action |

The checklist's compiled-artifact item concerns the submitted supplement, not
a requirement to commit machine-specific binaries to a source repository.
The journal guide also requires the manuscript's results to agree with the
submitted code. Cached rendering and three spot checks support that audit;
they do not justify marking a full original-input rerun or independent-seed
stability check as completed.
