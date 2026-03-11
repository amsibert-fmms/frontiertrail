"""WagonTrain world state class."""

from __future__ import annotations

import random
from enum import Enum
from typing import List


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

    GOAL_MILES = 2000
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

    # ------------------------------------------------------------------
    # State summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        river_str = " [RIVER AHEAD]" if self.river_ahead else ""
        return (
            f"Day {self.day:>3} | Miles: {self.miles_traveled:>6.1f}/{self.GOAL_MILES} | "
            f"Food: {self.food_supply:>5.1f} | Parts: {self.wagon_parts:>5.1f} | "
            f"Weather: {self.weather.value:<7} | Morale: {self.morale:>4.0f}{river_str}"
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
