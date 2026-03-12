# Terrain first-pass sweep summary

## Command

```bash
python tools/balancing_sweep.py --runs 30 --start-seed 1 --party-size 20 --output tools/reports/balancing_sweep_terrain_first_pass.csv
```

## Results (30 runs)

- Completion rate: **14/30 (46.7%)** reached Oregon City / Willamette Valley.
- Average day-to-finish among completed runs: **190.00**.
- Average survivors (all runs): **18.10**.
- Average survivors (completed runs): **18.86**.
- Average survivors (non-completed runs): **17.44**.
- Average end resources: food **188.42**, parts **65.57**, morale **54.71**.

## Notes

- This report is intended as a quick baseline after adding terrain-segment travel multipliers.
- Detailed per-seed output is in `balancing_sweep_terrain_first_pass.csv`.
