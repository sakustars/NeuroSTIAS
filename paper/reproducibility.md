# Reproducing all results

```bash
./install.sh full
source .venv/bin/activate
neurostias data fetch all          # downloads + checksums (local datasets must be provided)
bash benchmarks/envs/setup_baselines.sh   # isolated baseline environments
neurostias experiment all
neurostias report
```

Each experiment writes provenance.json (input SHA-256, git commit, parameters, seeds, package versions).
