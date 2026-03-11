# Chances, Formulas, and Outcome Review

This review focuses on the current probabilities/formulas in the simulation and their impact on outcomes.

## Quick health check

I ran a 300-seed Monte Carlo sweep of the full simulation loop (`Simulation(seed=i).run()` for `i=0..299`).

Observed outcomes:
- Win rate (reach 2000 miles): **0%**
- Average miles at end: **~516**
- Average end day: **~190**
- End reasons: mostly 200-day timeout, with some full-party deaths

## Current formulas and what they imply

## 1) Progress pacing is far too slow

### Current travel formula
- Base miles/day: `20`
- Weather multiplier: as low as `0.4` (stormy), `0.5` (snowy)
- Wagon parts multiplier: `min(1.0, wagon_parts/50.0)`
- Random factor: `0.8–1.2`

Even in decent conditions, expected daily travel hovers too low once weather + parts degradation + non-travel actions are included.

**Why this hurts outcomes:** reaching 2000 miles in 200 days requires ~10 miles/day average *across all days*. The current decision policy spends many days on rest/hunt/repair/river delays, so travel days must be much higher than 10 miles/day to compensate.

### Suggested improvement
- Raise `TRAVEL_BASE_MILES` from `20` to **`28–32`**.
- Make parts penalty less punishing by replacing `wagon_parts/50` with **`0.5 + wagon_parts/100`** (range 0.5..1.5, then clamp to 1.2 or 1.3 max if needed).
- Reduce daily forced drag slightly:
  - spoilage from `0.3–1.0` to **`0.2–0.7`**
  - wear from `0.0–1.0` to **`0.0–0.7`**

## 2) Event system is frequent and swingy

### Current event logic
- Daily event chance: `0.35`
- Uniform random from 15 events, including multiple high-penalty health events and large morale swings.

**Why this hurts outcomes:** event frequency is high enough that stacking penalties (health + morale + parts + food) often outpaces recovery options.

### Suggested improvement
- Lower `BASE_EVENT_CHANCE` from `0.35` to **`0.22–0.28`**.
- Split catalogue into weighted categories and tune weights instead of uniform choice:
  - neutral/flavor events: high weight
  - mild negative: medium
  - severe negative: low
  - positive: medium
- Add anti-streak protection (e.g., severe negative cannot occur 2 days in a row).

## 3) Hunting has very high success with 2 hunters

### Current hunt formula
- `success = min(0.95, 0.4 + 0.3 * num_hunters)`
- With 2 hunters in default party, success = **1.0 capped to 0.95**
- Reward on success: `30–70` food

**Why this is awkward:** survival still fails despite strong hunting, which indicates travel pacing + penalty pressure are dominant bottlenecks. Hunt can become a repetitive “resource reset” action without advancing journey goals.

### Suggested improvement
- Keep hunt useful but less binary:
  - success formula: **`0.3 + 0.2 * hunters`**, cap `0.85–0.9`
  - on failure, small consolation gain (e.g., `5–12` food)
- Optional: tie hunt yield to weather/terrain so bad weather reduces reliability.

## 4) River crossing risk may create dead time

### Current ford formula
- Risk: `max(0.1, 0.6 - 0.1 * scouts)`
- On failure: random health loss to everyone + food loss + morale hit
- Crossing always gives only `5` miles and consumes the day

**Why this hurts outcomes:** crossing day contributes little progress and can inflict large setbacks.

### Suggested improvement
- Keep tension but reduce stall effect:
  - increase `FORD_MILES` from `5` to **`8–12`**
  - cap group health loss severity (smaller upper bound)
  - add positive payoff for safe crossing (small parts/morale boost)

## 5) Decision policy is reactive but not strategic

The party mostly reacts to immediate low resources and doesn’t optimize for route progress over time.

### Suggested improvement
- Add a lightweight heuristic score per action each day:
  - `progress value + survival value + risk penalty`
- Include a “must-travel target” rule:
  - if projected miles/day needed to finish exceeds threshold, bias toward travel unless critical needs are red.

## Recommended tuning sequence (small safe steps)

1. **Pacing first**
   - Increase travel base + slightly soften wear/spoilage.
2. **Event pressure second**
   - Lower event frequency and weight severe negatives down.
3. **River and hunt cleanup**
   - Raise ford day progress; smooth hunt success/yields.
4. **Policy upgrade**
   - Add simple projected-finish heuristic.

This sequence should improve completion odds without removing survival pressure.

## Suggested target metrics after tuning

For 300-seed sweeps with default party:
- Win rate target: **25–45%**
- Average survivors on win: **4–8**
- Timeout rate: **<40%**
- Full-party death rate: **<20%**

Those targets keep challenge while making success realistically attainable.
