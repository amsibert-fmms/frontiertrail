# Balancing comparison: `EventSystem.BASE_EVENT_CHANCE` 0.22 -> 0.30

ELI5:
- We changed exactly one knob: "how often a random event happens each day."
- Everything else stayed the same.
- Then we re-ran the same 30 seeds and compared the final trend numbers.

## Protocol lock (same for both sweeps)
- Runs: 30
- Start seed: 1
- Party size: 20
- Baseline CSV: `tools/reports/balancing_sweep.csv`
- Tuned CSV: `tools/reports/balancing_sweep_event_chance_030.csv`

## Final trend deltas (tuned minus baseline)
- reach_rate_to_date: `-0.2000` (53.3% vs 73.3%)
- avg_day_to_finish_reached_to_date: `+1.2102` days
- avg_end_food_to_date: `-6.3144`
- avg_end_parts_to_date: `-10.3861`
- avg_end_morale_to_date: `+0.0749`

## Variability spot-check (population standard deviation across 30 rows)
- Survivor count standard deviation: `0.25 -> 1.08` (higher variability)
- End food standard deviation: `163.58 -> 140.20`
- End parts standard deviation: `33.62 -> 29.38`
- End morale standard deviation: `30.48 -> 29.08`

## Interpretation
- This one-knob change clearly increases uncertainty in *survivor outcomes* and lowers overall success rate.
- It also makes resources tighter on average (especially wagon parts).
- This is a stronger difficulty push, not just "more flavor variability."
