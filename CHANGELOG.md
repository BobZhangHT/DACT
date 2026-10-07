# Changelog

## Unreleased — prepared 2026-10-07

### Statistical implementation

- Use rank-based Monte Carlo cutoffs consistent with add-one p-values,
  including small simulation counts and ties.
- Validate array shapes, nonnegative finite counts, age scores, normalized
  delay masses, structural-zero compatibility and numerical settings.
- Improve fit-status reporting and bootstrap diagnostics in both numerical
  backends.
- Select the earliest admissible date when score maxima tie.

### Application and figures

- Clarify the absolute-score search plot, its model-based critical value,
  cumulative-information search range and selected date.
- Improve figure labels and use age-effect terminology in generated figures
  and table captions.
- Retain the full-year 60/90/120-day delay-support comparison. Its metadata
  now hash the actual Parquet inputs directly.

### Documentation and tests

- Align the README and citation title with the current manuscript; document
  setup, data requirements, result interpretation and Git upload scope.
- Add numerical validation, calibration, bootstrap diagnostics, tie-resolution
  and input-fingerprint checks.
- Keep existing function names and the `gradient_ratio` output key for API
  compatibility; document their age-effect meaning.
