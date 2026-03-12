# Chances, Formulas, and Outcome Review (Current Build)

This review re-checks the **current** simulation formulas and probabilities using fresh Monte Carlo runs, then suggests targeted tuning changes.

## Outcome snapshot (what happens right now)

I ran two sweeps:

1. **500 full runs** (`Simulation(seed=i).run()` for `i=0..499`) to measure win/survival outcomes.
2. **300 instrumented runs** (manual daily loop) to measure event frequency and action mix.

Observed outcomes:

- Win rate: **22.4%** (112 / 500)
- Average end miles: **1515.9**
- Median end miles: **1484.0**
- Average run length: **165.5 days**
- Average survivors on wins: **6.78**
- End states (300-run instrumented sweep):
  - **Win: 80**
  - **Timeout: 150**
  - **All dead: 70**

Interpretation: the game is now *winnable* (good), but still leans punishing, with many runs timing out or collapsing late.

---

## Formula and chance review

## 1) Event pressure is calibrated reasonably, but severe health shocks still snowball

### Current formulas
- Daily event chance: `BASE_EVENT_CHANCE = 0.22`
- Weighted selection by severity, with anti-streak suppression for back-to-back severe negatives.

### Measured outcome
- Realized event-day rate over 300 runs: **21.9%** (very close to target).

### Improvement suggestions
- Keep event rate at `0.22`.
- Soften *group-wide* health hits on severe disease events (e.g., reduce `-20` to `-12..-16` on cholera/dysentery style events).
- Keep single-target danger events (snake bite) for drama, but reduce frequency of events that damage **everyone at once**.

Why: wipe-style health attrition is a major driver of all-dead endings.

## 2) River crossing risk is still too costly for how often it appears

### Current formulas
- Ford risk: `max(0.1, 0.6 - 0.1 * scouts)`.
- Default party has 1 scout ⇒ failure risk near **0.5**.
- Failed ford applies random per-agent health loss + food loss + morale hit.

### Measured outcome
- Observed ford failure rate: **50.8%** (1692 attempts).

### Improvement suggestions
- Shift risk curve down slightly (example: `0.5 - 0.1 * scouts`, floor `0.08`).
- Narrow failure health loss band (example: `3..10` instead of `4..15`).
- Add a small positive reward for safe crossing beyond morale (already includes parts +2, which is good).

Why: river outcomes currently feel close to coin flips with very asymmetric downside.

## 3) Action selection over-indexes on maintenance actions

### Measured action mix (300-run instrumented sweep)
- Hunt: **15,931**
- Travel: **15,728**
- Rest: **15,435**
- Ford: **1,692**
- Ration: **447**
- Repair: **272**

### Interpretation
The top 3 actions are almost evenly split. That means progress actions are not dominating enough to consistently hit 2000 miles by day 200, which explains the high timeout count.

### Improvement suggestions
- Increase schedule-pressure bias slightly in decision scoring when behind pace.
- Introduce a mild “consecutive non-travel day” penalty to action scores (except during critical survival states).
- Consider small passive recovery on travel days for high-morale parties, so travel does not always feel strictly worse than rest/hunt loops.

## 4) Travel pacing is close, but still vulnerable to drift

### Current formulas
- `TRAVEL_BASE_MILES = 32`
- Weather and wagon-parts multipliers can push daily progress down heavily in bad conditions.

### Improvement suggestions
- Keep base travel at 32 for now.
- Add a tiny guaranteed floor on travel-day miles (example: minimum 8–10 miles unless blocked by river).
- Optionally reduce extreme weather drag (`stormy` from `0.4` to `0.45`) if timeout rate remains high after decision-tuning.

Why: many near-miss runs likely fail because cumulative bad-weather sequences erase progress pace.

---

## Recommended tuning order (small, safe iterations)

1. **River risk softening** (highest pain for least design disruption).
2. **Decision-policy nudge toward progress** when behind schedule.
3. **Severe group-health event softening** to reduce full-party wipeouts.
4. Re-run 500-seed sweep and compare against targets.

## Updated target metrics

After the next tuning pass, aim for:

- Win rate: **30–40%**
- Timeout rate: **<45%**
- All-dead rate: **<15–18%**
- Survivors on wins: **5–8 average**

These targets should preserve challenge while reducing “unrecoverable spiral” runs.

---

## Implemented improvements in this pass

- Added a startup prompt for party size when running `main.py` (with a `--party-size` override for non-interactive use).
- Wagon-break events now assign an urgent repair owner (prefers mechanic), so repair pressure is explicitly represented in decisions.
- Added day-by-day low-progress streak tracking and fed that into agent decision scoring so travel/ford urgency grows when the group stalls.
- Removed backward movement behavior from negative mileage events (no event can subtract miles traveled anymore).
- Increased morale sensitivity to stalling, hunger, poor health, poor wagon condition, and adverse weather.

## Next suggestions after this implementation

- Re-run 500-seed and 1000-seed sweeps to quantify the effect of stronger morale pressure on death vs timeout outcomes.
- If death rate rises too much, soften only health-linked morale penalty before touching travel pace.
- Consider one role-aware mitigation: medic can reduce morale loss from sickness-related event days.
