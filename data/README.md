# Data

Nothing in this folder except `manifest.yaml`, `checksums.lock.json` and this README is committed.

```bash
neurostias data list            # what is declared and what is present
neurostias data fetch <id>      # download one dataset (or `all`)
neurostias data verify          # re-check SHA-256 of downloaded files
```

Every dataset entry records its source, licence and citation. Results from
semi-synthetic tests (real data with a known perturbation applied) are labelled
`semi_synthetic` in their provenance and in every report table.
