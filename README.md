# Frontier Trail

An environment for AI agents on the Oregon Trail.

## Historical Trail Background

The country that lay ahead of the pioneers contained no towns or permanent settlements. For weeks, emigrants crossed vast grasslands that were hot by day and cold at night. Violent thunderstorms often swept down on travelers. Eventually, they crossed the snow-capped Rocky Mountains. Beyond the mountains lay a harsh wilderness of sagebrush desert, canyons, and forests.

The trail began at old Independence Landing, north of Independence, Missouri. Emigrants left steamboats here after a five- or six-day journey from St. Louis. The center of activity in Independence was the bustling town square. Most pioneers camped a mile or two away while purchasing supplies for a four- to five-month trek.

Travel parties with horses or mules often departed first so their animals could graze the shorter grasses. The majority, using powerful and durable oxen, left about two weeks later.

Nothing contributed more to success or failure on a western wagon trek than the wagons themselves. Pioneers needed wagons strong enough to haul families and supplies for five months or more. To withstand the rugged trail, wagons were commonly built from seasoned hardwood. Most emigrants used modified farm wagons with canvas covers stretched over hooped frames.

A family of four could travel with a single wagon, but space was extremely tight because supplies occupied nearly the entire interior. Families who could afford it often brought more than one wagon. Some emigrants adapted existing farm wagons for the trip; others purchased rigs built specifically for a one-way journey west.

A wagon had to be light enough to avoid overtaxing mules or oxen and strong enough to avoid breaking under loads as heavy as 2,500 pounds. For this reason, wagons were constructed from hardwoods such as maple, hickory, and oak. Iron was used mainly to reinforce high-stress components such as tires, axles, and hounds.

Even then, emigrant wagons were uncomfortable. Most lacked springs, and there was little room to sit inside because cargo took priority.

## What currently affects outcomes in this simulator

If you're thinking in terms of "scoring," this project currently uses outcome metrics rather than one single score value.

- **Primary success condition:** reach the selected destination before the day limit.
- **Other tracked outcomes:** survivors, days used, miles traveled, food remaining, wagon parts, and morale.

### Your specific questions, mapped to current behavior

1. **Earlier parties (horses/mules) vs later parties (oxen):**
   - Yes, this can affect outcomes.
   - The simulator tracks oxen, horses, and mules separately and applies weighted draft power to travel speed.
   - Losing animals lowers mobility (with a floor so movement does not collapse to zero).

2. **Terrain/travel speed by segment:**
   - **Partially modeled.** Weather directly modifies travel speed and river crossings can alter risk/progress.
   - Route choice after Soda Springs changes destination distance.
   - There is **not yet** a dedicated per-segment terrain-speed table (for example, explicit "plains vs Rockies vs desert" multipliers by map segment).

3. **How many wagons:**
   - The simulation currently abstracts to one shared wagon-state model (`wagon_parts`) rather than modeling multiple independent wagons.

4. **Lost livestock => lost wagon / less supply capacity:**
   - Losing livestock reduces movement speed.
   - Starvation logic can slaughter draft animals for emergency food, which helps short-term survival but hurts future travel power.
   - Animal loss does **not** currently delete a wagon entity or dynamically reduce cargo-capacity limits, because per-wagon capacity is not explicitly simulated.

## Suggested next balancing expansions

If you want historical realism to impact outcomes more directly, high-value next steps are:

- Add explicit **terrain segment modifiers** (plains, mountains, desert) that stack with weather.
- Add **wagon count and cargo-capacity mechanics** so extra wagons improve carrying capacity but increase maintenance burden.
- Add **hard mobility thresholds** where severe animal loss can force stop-days, offloading, or abandonment choices.
