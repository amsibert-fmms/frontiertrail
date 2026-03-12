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

---

## Formula audit: what looks off right now

### 1) Hunt expected value is likely too strong vs travel

Current hunt success formula (`0.3 + 0.2 * hunters`, capped at `0.9`, then weather multiplier) can make hunt very reliable with even a small number of hunters. On failure, the action still grants `5–12` food, so downside is muted. This can keep action mix sticky around hunt-heavy loops, especially under schedule pressure where the system is trying to increase travel but not reducing hunt enough.

**Suggestion:** reduce guaranteed-forage floor and flatten hunter scaling.

- Success chance: `0.28 + 0.16 * hunters` (cap `0.82`) before weather.
- Failed-hunt forage: reduce to `3–8`.
- Add diminishing return after 2 hunters (e.g., extra hunters add `+0.08` each).

Why this is safer than blunt nerfs: it preserves hunter identity while reducing near-auto-success patterns that can crowd out progress actions.

### 2) Ford risk is static and under-reacts to context

Current ford risk is mostly role-count based (scouts) and does not account for weather/condition at crossing time. This creates edge cases where bad-weather crossings are not meaningfully scarier than favorable ones.

**Suggestion:** make risk context-aware with bounded modifiers.

- Start with current base/floor model.
- Add weather modifier: `+0.00` sunny/cloudy, `+0.05` rainy, `+0.10` snowy, `+0.15` stormy.
- Add wagon-parts modifier when badly damaged (e.g., `+0.08` if parts `< 30`).
- Keep hard cap at `0.75` to avoid hopeless states.

This should improve decision signal quality: waiting/resting before dangerous crossings becomes rational in bad conditions.

### 3) Morale penalties may be compounding too aggressively in weak states

`_update_morale` applies multiple additive penalties every day (progress, hunger, health, mechanical, weather). In low-resource streaks, this can drive morale collapse quickly and amplify all-dead outcomes.

**Suggestion:** use soft caps for stacked penalties.

- Cap total daily world-pressure penalty to `<= 7.0`.
- Scale progress penalty by health state (if avg health `< 45`, apply only `70%` of progress penalty).
- Keep weather penalty uncapped inside the total cap to preserve atmosphere.

This keeps pressure meaningful without runaway cascades.

### 4) Progress-bias currently trades some timeouts for deaths

Pass #2 reduced timeout slightly but raised all-dead rate. That pattern usually indicates catch-up pressure is entering already fragile states.

**Suggestion:** couple progress pressure to survivability margin.

- Compute margin score from avg health, food days remaining, and wagon parts.
- When margin is low, attenuate non-travel pressure by `20–35%`.
- Keep travel bias unchanged when margin is healthy.

This retains pace gains while reducing lethal overcommitment.

---

## Concrete experiment plan (recommended)

Run these as separate, additive A/B passes using the same seed ranges:

1. **Pass 3A (hunt rebalance only)**
2. **Pass 3B (ford context risk only)**
3. **Pass 3C (morale soft-cap + survivability-gated pressure)**
4. **Pass 3D (all combined, with only half-strength coefficients first)**

Track these acceptance thresholds before keeping changes:

- Win rate: **+1.5 pp or better** vs pass #2.
- Timeout: **not worse** than pass #2 by more than `+0.5 pp`.
- All-dead: **improves by at least -1.0 pp**.
- Median miles: non-decreasing.

If 3D misses targets, keep only the best single-factor pass (likely 3C) and re-tune in smaller increments.

---

## Pass 3A implementation + measured outcome

Implemented hunt-only rebalance in `wagon_train/decisions.py` using constants (no ford/morale/progress-pressure changes):

- `HUNT_SUCCESS_BASE = 0.30`
- `HUNT_SUCCESS_PER_HUNTER = 0.18`
- `HUNT_SUCCESS_EXTRA_HUNTER = 0.10` (after first two hunters)
- `HUNT_SUCCESS_CAP = 0.86`
- failed-hunt forage reduced to `4–10`

### 500-run outcomes (seed `0..499`)

| Metric | Pass #2 target guardrail | Pass 3A result | Pass/Fail |
|---|---:|---:|---:|
| Win rate | `>= 28.9%` | `21.8%` | ❌ |
| Timeout rate | `<= 52.3%` | `41.0%` | ✅ |
| All-dead rate | `<= 19.8%` | `37.2%` | ❌ |
| Median end miles | `>= 1602.5` | `1551.5` | ❌ |

### 300-run instrumented outcomes (seed `10000..10299`)

- Event-day rate: `22.18%`
- Ford attempts: `1660`
- Ford failures: `700`
- Ford failure rate: `42.17%`

Action mix snapshot:

- Travel: `15310`
- Hunt: `14909`
- Rest: `12931`
- Ford: `1660`
- Repair: `279`
- Ration: `447`

### Conclusion

Pass 3A **did not meet acceptance thresholds**. While timeout rate improved, survivability and completion quality regressed (all-dead increased and median miles dropped below guardrail). Recommended next move: either revert 3A coefficients toward pass #2 behavior or pair hunt changes with the Pass 3C safety valves before re-evaluating.

---

## Pass 3A+ (dated logs + Sunday-rest archetype) implementation + measured outcome

### What changed

Implemented in current code:

- **Dated log headers** now start at **Mar 1** with a random year between **1840 and 1860** for each run.
- Added a **religious archetype** role (`preacher`) in the default party.
- Preacher behavior: on Sundays, the preacher prefers `REST` when the party is not in a critical survival emergency.
- Sunday work rule: if the party performs a non-rest action on Sunday, work is still allowed, but a **half-rest penalty** is applied (small extra health/morale drain).

### Dated log sample

Observed in a sample run (`seed=42`):

- `Day   1 (Mar 01, 1843)`
- `Day   2 (Mar 02, 1843)`
- `Day   3 (Mar 03, 1843)`

This confirms the requested date formatting and March 1 start behavior.

### 500-run outcomes (seed `0..499`)

| Metric | Pass #2 target guardrail | Pass 3A+ result | Pass/Fail |
|---|---:|---:|---:|
| Win rate | `>= 28.9%` | `18.8%` | ❌ |
| Timeout rate | `<= 52.3%` | `46.0%` | ✅ |
| All-dead rate | `<= 19.8%` | `35.2%` | ❌ |
| Median end miles | `>= 1602.5` | `1439.1` | ❌ |

Additional 500-run stats:

- Avg end miles: `1510.0`
- Avg run length: `155.6` days
- Avg survivors on wins: `4.22`

### 300-run instrumented outcomes (seed `10000..10299`)

- Event-day rate: `21.72%`
- Ford attempts (river present): `1663`
- Ford failures: `672`
- Ford failure rate: `40.41%`
- Sunday work days observed: `4429`

Action mix snapshot:

- Travel: `14863`
- Hunt: `13780`
- Rest: `14175`
- Ford: `1663`
- Repair: `297`
- Ration: `292`

### Interpretation and pushback

This change set improves **timeouts** but is still unsafe for overall outcomes: **win rate and survivability remain far below guardrails**. In plain terms, this is likely too punishing when combined with existing pressure systems.

Pushback: keeping Sunday penalties at this strength without compensating safety valves is probably a bad idea if your primary objective is higher completion and lower mortality.

### Suggested improvements (next safe iteration)

1. **Soften Sunday work penalty by ~35–45%**
   - Keep flavor, reduce survivability tax.
   - Example: health loss `1.0–2.5`, morale loss `0.5–2.0`.
2. **Gate Sunday penalty by condition**
   - If avg health `< 45` or food `< 20`, skip penalty entirely (already under emergency stress).
3. **Restrict preacher hard-rest behavior to non-river days**
   - On Sundays with river ahead, allow preacher to support ford/travel if progress is critically behind schedule.
4. **Pair with Pass 3C safety valves**
   - Morale pressure soft cap + survivability-gated non-travel pressure before judging Sunday mechanics as final.

