"""WagonTrain world state class."""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Trail map configuration
# ---------------------------------------------------------------------------
# ELI5: this is the "trip checklist" in order from start to finish.
# Each stop has its cumulative miles from Independence.
#
# Why cumulative miles?
# - It makes "How far have we gone?" comparisons super easy.
# - It also makes the finish-line math explicit and readable.
# - We can still derive per-segment distances when we need them.
TRAIL_STOPS: List[dict[str, object]] = [
    {"name": "Independence, Missouri", "distance": 0},
    {"name": "Kansas River Crossing", "distance": 102, "river": True},
    {"name": "Big Blue River Crossing", "distance": 185, "river": True},
    {"name": "Fort Kearny", "distance": 320, "fort": True},
    {"name": "Platte River Valley", "distance": 350},
    {"name": "Ash Hollow", "distance": 460},
    {"name": "Chimney Rock", "distance": 520, "landmark": True},
    {"name": "Scotts Bluff", "distance": 560, "landmark": True},
    {"name": "Fort Laramie", "distance": 640, "fort": True},
    {"name": "North Platte River Crossing", "distance": 700, "river": True},
    {"name": "Register Cliff", "distance": 750},
    {"name": "Independence Rock", "distance": 830, "landmark": True},
    {"name": "Devil's Gate", "distance": 850, "landmark": True},
    {"name": "South Pass (Continental Divide)", "distance": 932, "landmark": True},
    {"name": "Green River Crossing", "distance": 989, "river": True},
    {"name": "Fort Bridger", "distance": 1020, "fort": True},
    {"name": "Soda Springs", "distance": 1180},
    {"name": "Fort Hall", "distance": 1300, "fort": True},
    {"name": "Snake River Crossing", "distance": 1350, "river": True},
    {"name": "Fort Boise", "distance": 1520, "fort": True},
    {"name": "Grande Ronde Valley", "distance": 1650},
    {"name": "Blue Mountains", "distance": 1700, "landmark": True},
    {"name": "Fort Walla Walla", "distance": 1740, "fort": True},
    {"name": "Columbia River", "distance": 1850, "river": True},
    {"name": "The Dalles", "distance": 1900},
    {"name": "Barlow Road", "distance": 1980},
    {"name": "Oregon City / Willamette Valley", "distance": 2040},
]


def _build_landmark_graph_from_stops() -> Dict[str, Dict[str, List[Tuple[str, int]] | int]]:
    """Build a single-path landmark graph from the ordered stop list.

    ELI5: we convert the checklist into linked "next stop" hops,
    where each hop stores the miles between neighboring stops.
    """
    graph: Dict[str, Dict[str, List[Tuple[str, int]] | int]] = {}
    for index, stop in enumerate(TRAIL_STOPS):
        name = str(stop["name"])
        distance = int(stop["distance"])

        # Last stop has no "next" hop.
        if index == len(TRAIL_STOPS) - 1:
            graph[name] = {"distance": distance, "next": []}
            continue

        next_stop = TRAIL_STOPS[index + 1]
        next_name = str(next_stop["name"])
        next_distance = int(next_stop["distance"])

        # Segment miles are computed from cumulative-mile entries.
        segment_miles = next_distance - distance
        graph[name] = {"distance": distance, "next": [(next_name, segment_miles)]}

    return graph


# This mirrors the original graph-style format, but now it is generated from
# the ordered source-of-truth stop list above.
LANDMARKS = _build_landmark_graph_from_stops()


def _distance_to_destination(start: str, destination: str) -> int:
    """Return the remaining miles between two nodes on the configured trail."""
    if start == destination:
        return 0

    next_steps = LANDMARKS[start].get("next", [])
    if not next_steps:
        # ELI5: no next step means this node cannot reach destination.
        return 10**9

    next_name, miles = next_steps[0]
    return miles + _distance_to_destination(next_name, destination)


# Start/end for this simulation run.
TRAIL_START = str(TRAIL_STOPS[0]["name"])
TRAIL_DESTINATION = str(TRAIL_STOPS[-1]["name"])

# Quick-lookup helpers derived from TRAIL_STOPS.
# ELI5: these are like indexes in a book so we can find stop details fast.
STOP_BY_NAME: Dict[str, dict[str, object]] = {
    str(stop["name"]): stop for stop in TRAIL_STOPS
}


class Weather(str, Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"
    HOT = "hot"


# Travel-miles modifier per weather type
WEATHER_TRAVEL_MODIFIER: dict[Weather, float] = {
    Weather.SUNNY: 1.0,
    Weather.CLOUDY: 0.95,
    Weather.RAINY: 0.75,
    Weather.STORMY: 0.4,
    Weather.SNOWY: 0.5,
    Weather.HOT: 0.85,
}

# Probability of transitioning to each weather state (simple Markov weights)
_WEATHER_TRANSITIONS: dict[Weather, dict[Weather, float]] = {
    Weather.SUNNY: {
        Weather.SUNNY: 0.5, Weather.CLOUDY: 0.25, Weather.RAINY: 0.1,
        Weather.HOT: 0.1, Weather.STORMY: 0.03, Weather.SNOWY: 0.02,
    },
    Weather.CLOUDY: {
        Weather.SUNNY: 0.3, Weather.CLOUDY: 0.3, Weather.RAINY: 0.25,
        Weather.HOT: 0.05, Weather.STORMY: 0.07, Weather.SNOWY: 0.03,
    },
    Weather.RAINY: {
        Weather.SUNNY: 0.2, Weather.CLOUDY: 0.35, Weather.RAINY: 0.25,
        Weather.HOT: 0.02, Weather.STORMY: 0.15, Weather.SNOWY: 0.03,
    },
    Weather.STORMY: {
        Weather.SUNNY: 0.1, Weather.CLOUDY: 0.3, Weather.RAINY: 0.4,
        Weather.HOT: 0.02, Weather.STORMY: 0.15, Weather.SNOWY: 0.03,
    },
    Weather.SNOWY: {
        Weather.SUNNY: 0.15, Weather.CLOUDY: 0.3, Weather.RAINY: 0.1,
        Weather.HOT: 0.0, Weather.STORMY: 0.05, Weather.SNOWY: 0.4,
    },
    Weather.HOT: {
        Weather.SUNNY: 0.4, Weather.CLOUDY: 0.2, Weather.RAINY: 0.1,
        Weather.HOT: 0.25, Weather.STORMY: 0.04, Weather.SNOWY: 0.01,
    },
}


def _weighted_choice(weights: dict) -> Weather:
    items = list(weights.items())
    keys, vals = zip(*items)
    return random.choices(keys, weights=vals, k=1)[0]


class WagonTrain:
    """Holds the shared world state for the wagon train simulation."""

    # Total Oregon-route distance derived from the landmark graph.
    GOAL_MILES = _distance_to_destination(TRAIL_START, TRAIL_DESTINATION)
    MAX_DAYS = 200
    # This is a small quality-of-life helper for planning decisions:
    # if we know our goal and max days, we can estimate whether we are
    # "on pace" or "behind pace" as the trip unfolds.
    REQUIRED_MILES_PER_DAY = GOAL_MILES / MAX_DAYS

    def __init__(
        self,
        food_supply: float = 500.0,
        wagon_parts: float = 100.0,
        weather: Weather = Weather.SUNNY,
    ) -> None:
        self.day: int = 0
        self.miles_traveled: float = 0.0
        self.food_supply: float = food_supply
        self.wagon_parts: float = wagon_parts
        self.weather: Weather = weather
        self.morale: float = 80.0          # group morale (0-100)
        self.sickness_events: List[str] = []
        # Track who (if anyone) is urgently responsible for fixing wagon damage.
        # This is set by severe breakage events and cleared after a repair action.
        self.urgent_repair_assignee: str | None = None
        # Keep lightweight day-to-day pacing memory so agents can react when
        # progress has stalled for several days in a row.
        self.low_progress_streak: int = 0
        # Expose today's progress for logging/tuning; this is never negative.
        self.last_daily_progress: float = 0.0
        self.river_ahead: bool = False
        self._days_until_river: int = random.randint(5, 15)
        self._event_log: List[str] = []    # events that happened today

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def sickness_count(self) -> int:
        return len(self.sickness_events)

    @property
    def travel_modifier(self) -> float:
        return WEATHER_TRAVEL_MODIFIER[self.weather]

    @property
    def is_finished(self) -> bool:
        return self.miles_traveled >= self.GOAL_MILES or self.day >= self.MAX_DAYS

    @property
    def next_landmark(self) -> str:
        """Return the next named stop ahead of current progress.

        ELI5:
        - Look at each stop's cumulative mile marker.
        - Find the first one that is still ahead of us.
        - If we are already at/after the end, keep returning destination.
        """
        for stop in TRAIL_STOPS:
            stop_distance = int(stop["distance"])
            if self.miles_traveled < stop_distance:
                return str(stop["name"])
        return TRAIL_DESTINATION

    @property
    def current_or_last_stop(self) -> str:
        """Return the furthest stop reached at current mileage.

        ELI5:
        - We walk through the ordered trail stops.
        - Keep remembering the latest stop we have reached.
        - This lets game systems ask, "Where are we right now-ish?"
        """
        current_name = TRAIL_START
        for stop in TRAIL_STOPS:
            stop_name = str(stop["name"])
            stop_distance = int(stop["distance"])
            if self.miles_traveled >= stop_distance:
                current_name = stop_name
            else:
                break
        return current_name

    def _stop_has_tag(self, stop_name: str, tag: str) -> bool:
        """Return whether a named stop carries the given metadata tag.

        ELI5: tags are tiny labels like "fort" or "river" attached to stops.
        """
        stop = STOP_BY_NAME.get(stop_name)
        if stop is None:
            return False
        return bool(stop.get(tag, False))

    def is_fort_stop(self, stop_name: str) -> bool:
        """Return True if stop_name is a fort stop."""
        return self._stop_has_tag(stop_name, "fort")

    def is_river_crossing_stop(self, stop_name: str) -> bool:
        """Return True if stop_name is tagged as a river crossing stop."""
        return self._stop_has_tag(stop_name, "river")

    def is_landmark_stop(self, stop_name: str) -> bool:
        """Return True if stop_name is tagged as a narrative landmark."""
        return self._stop_has_tag(stop_name, "landmark")

    @property
    def at_fort_stop(self) -> bool:
        """Return True when current location corresponds to a fort stop."""
        return self.is_fort_stop(self.current_or_last_stop)

    @property
    def at_river_crossing_stop(self) -> bool:
        """Return True when current location corresponds to a river crossing stop."""
        return self.is_river_crossing_stop(self.current_or_last_stop)

    @property
    def at_landmark_stop(self) -> bool:
        """Return True when current location corresponds to a landmark stop."""
        return self.is_landmark_stop(self.current_or_last_stop)

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------

    def advance_day(self) -> None:
        """Tick the calendar and update persistent world state."""
        self.day += 1
        self._event_log = []

        # Evolve weather
        self.weather = _weighted_choice(_WEATHER_TRANSITIONS[self.weather])

        # River management
        self._days_until_river -= 1
        self.river_ahead = self._days_until_river <= 0

        # Natural food spoilage each day.
        # We softened this range so the party is still under pressure,
        # but the simulation is less dominated by unavoidable attrition.
        spoilage = random.uniform(0.2, 0.7)
        self.food_supply = max(0.0, self.food_supply - spoilage)

        # Wagon wear from terrain and normal use.
        # This was also softened to reduce extreme compounding slowdowns.
        wear = random.uniform(0.0, 0.7)
        self.wagon_parts = max(0.0, self.wagon_parts - wear)

        # Clear today's sickness events (illness resolved or carried over)
        self.sickness_events = []

    def river_crossed(self) -> None:
        """Reset river state after the party fords it."""
        self.river_ahead = False
        self._days_until_river = random.randint(8, 20)

    def record_daily_progress(self, miles_gained_today: float) -> None:
        """Record daily progress and update the low-progress streak.

        ELI5 version:
        - If we moved a solid amount today, reset the "stuck" counter.
        - If we barely moved (or did not move), increase the stuck counter.
        - We never treat progress as negative because the wagon cannot travel
          backwards toward the start.
        """
        # Safety clamp: do not let any caller record negative travel progress.
        self.last_daily_progress = max(0.0, miles_gained_today)

        # "Low progress" threshold is intentionally small: if we are under this
        # amount for multiple days, decision logic should prioritize moving.
        low_progress_threshold = 5.0
        if self.last_daily_progress < low_progress_threshold:
            self.low_progress_streak += 1
        else:
            self.low_progress_streak = 0

    # ------------------------------------------------------------------
    # State summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        river_str = " [RIVER AHEAD]" if self.river_ahead else ""
        return (
            f"Day {self.day:>3} | Miles: {self.miles_traveled:>6.1f}/{self.GOAL_MILES} | "
            f"Food: {self.food_supply:>5.1f} | Parts: {self.wagon_parts:>5.1f} | "
            f"Weather: {self.weather.value:<7} | Morale: {self.morale:>4.0f} | "
            f"Next: {self.next_landmark}{river_str}"
        )

    @property
    def miles_needed_per_remaining_day(self) -> float:
        """Return the pace needed from now to reach the goal in time.

        Think of this like a "speedometer target": if this value gets high,
        the group should prioritize progress unless survival is in immediate danger.
        """
        remaining_days = max(1, self.MAX_DAYS - self.day)
        remaining_miles = max(0.0, self.GOAL_MILES - self.miles_traveled)
        return remaining_miles / remaining_days

    def __repr__(self) -> str:
        return (
            f"WagonTrain(day={self.day}, miles={self.miles_traveled:.1f}, "
            f"food={self.food_supply:.1f}, parts={self.wagon_parts:.1f}, "
            f"weather={self.weather.value})"
        )
