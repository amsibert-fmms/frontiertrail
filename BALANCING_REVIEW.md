# Chances, Formulas, and Outcome Review (Current Build)

This document now includes **tuning pass #2 (decision-policy progress pressure)** and compares pass #1 vs pass #2 using the same seeds/harness.

## What was changed in pass #2

Implemented in `wagon_train/agent.py`:

- Increased progress weighting in `_action_score` for `TRAVEL` and `FORD_RIVER`.
- Added bounded **non-travel pressure** when behind schedule, which slightly lowers scores for `REST`, `HUNT`, `REPAIR_WAGON`, and `RATION_FOOD`.
- Increased stall sensitivity (`low_progress_streak`) in action scoring.
- Lowered progress-bias trigger threshold in `_apply_progress_bias` so catch-up mode engages earlier.

Design intent: reduce timeout drift without disabling survival actions in critical states.

---

## Measurement method (unchanged)

For both baseline (pass #1 code) and pass #2 code, I ran:

1. **500 full runs** (`Simulation(seed=i).run()` for `i=0..499`).
2. **300 instrumented runs** (`seed=10000..10299`) with counters on:
   - `EventSystem.roll`
   - `DecisionEngine.resolve`
   - `DecisionEngine.apply_action`

---

## Pass #1 (baseline) vs Pass #2

### 500-run outcomes

| Metric | Pass #1 baseline | Pass #2 | Delta |
|---|---:|---:|---:|
| Win rate | 27.6% | 27.4% | -0.2 pp |
| Timeout rate | 52.6% | 51.8% | -0.8 pp |
| All-dead rate | 19.8% | 20.8% | +1.0 pp |
| Avg end miles | 1590.1 | 1592.7 | +2.6 |
| Median end miles | 1594.5 | 1602.5 | +8.0 |
| Avg run length (days) | 164.3 | 163.6 | -0.6 |
| Avg survivors on wins | 6.95 | 6.95 | ~0 |

### 300-run instrumented outcomes

| Metric | Pass #1 baseline | Pass #2 | Delta |
|---|---:|---:|---:|
| Event-day rate | 21.79% | 21.76% | -0.03 pp |
| Ford attempts | 1674 | 1703 | +29 |
| Ford failures | 660 | 675 | +15 |
| Ford failure rate | 39.43% | 39.64% | +0.21 pp |

Action mix (pass #1 → pass #2):

- Travel: `15671 -> 15842` (up)
- Hunt: `15958 -> 15985` (flat)
- Rest: `14852 -> 14673` (down)
- Ford: `1674 -> 1703` (up)
- Repair: `254 -> 250` (slightly down)
- Ration: `483 -> 433` (down)

---

## Interpretation

What improved:

- Progress behavior moved in the intended direction (more travel/ford, less rest).
- Timeout rate and median miles improved slightly.

What did not improve enough:

- Win rate remained effectively flat.
- All-dead rate rose by ~1 percentage point.

Risk note:

- This suggests pass #2 added pace pressure successfully, but likely pushed some runs into lower-margin survival states. It is **not** a good idea to keep increasing pressure blindly; that may trade timeouts for deaths without raising wins.

---

## Recommended next pass (safe, constrained)

1. Keep pass #1 + pass #2 changes.
2. Add a **small travel-day minimum progress floor** on non-river travel days (e.g., 8–10 miles).
3. Add a tiny safety valve for high-risk survival states (e.g., if avg health is low, dampen non-travel pressure slightly).
4. Re-run the same 500/300 harness for direct comparison.
