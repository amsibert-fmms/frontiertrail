"""Random event system for the wagon train simulator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .agent import Agent
    from .world import WagonTrain


@dataclass
class Event:
    """A random event that may occur during the journey."""

    name: str
    description: str
    effects: dict  # keys: food_delta, parts_delta, health_delta, morale_delta, etc.


# ---------------------------------------------------------------------------
# Event catalogue
# ---------------------------------------------------------------------------

EVENT_CATALOGUE: List[dict] = [
    {
        "name": "Broken Axle",
        "description": "A wagon axle snaps under the strain of rough terrain.",
        "food_delta": 0,
        "parts_delta": -20,
        "health_delta": 0,
        "morale_delta": -5,
        "sickness": False,
    },
    {
        "name": "Dysentery Outbreak",
        "description": "Several party members fall ill with dysentery.",
        "food_delta": 0,
        "parts_delta": 0,
        "health_delta": -15,
        "morale_delta": -10,
        "sickness": True,
    },
    {
        "name": "Bountiful Prairie",
        "description": "The party crosses a lush prairie teeming with game and wild berries.",
        "food_delta": 30,
        "parts_delta": 0,
        "health_delta": 5,
        "morale_delta": 8,
        "sickness": False,
    },
    {
        "name": "Thunderstorm",
        "description": "A violent thunderstorm batters the wagon train overnight.",
        "food_delta": -10,
        "parts_delta": -10,
        "health_delta": -5,
        "morale_delta": -8,
        "sickness": False,
    },
    {
        "name": "Friendly Native Traders",
        "description": "A group of friendly natives offers to trade food for tools.",
        "food_delta": 20,
        "parts_delta": -5,
        "health_delta": 0,
        "morale_delta": 5,
        "sickness": False,
    },
    {
        "name": "Injured Animal",
        "description": "One of the draft animals is injured, slowing the wagon.",
        "food_delta": 0,
        "parts_delta": 0,
        "health_delta": 0,
        "morale_delta": -6,
        "sickness": False,
        "miles_delta": -5,
    },
    {
        "name": "Abandoned Cache",
        "description": "The party discovers an abandoned cache of supplies.",
        "food_delta": 15,
        "parts_delta": 10,
        "health_delta": 0,
        "morale_delta": 10,
        "sickness": False,
    },
    {
        "name": "Snake Bite",
        "description": "A traveler is bitten by a rattlesnake.",
        "food_delta": 0,
        "parts_delta": 0,
        "health_delta": -20,
        "morale_delta": -8,
        "sickness": True,
        "target": "random",
    },
    {
        "name": "Hailstorm",
        "description": "Large hailstones damage the wagon canvas and frighten the animals.",
        "food_delta": -5,
        "parts_delta": -8,
        "health_delta": -3,
        "morale_delta": -5,
        "sickness": False,
    },
    {
        "name": "Beautiful Sunset",
        "description": "The evening sky blazes with colour, lifting everyone's spirits.",
        "food_delta": 0,
        "parts_delta": 0,
        "health_delta": 0,
        "morale_delta": 12,
        "sickness": False,
    },
    {
        "name": "Rotten Food",
        "description": "A portion of the food stores has spoiled.",
        "food_delta": -20,
        "parts_delta": 0,
        "health_delta": 0,
        "morale_delta": -5,
        "sickness": False,
    },
    {
        "name": "Rough River Crossing",
        "description": "The wagon nearly tips during a river crossing.",
        "food_delta": -10,
        "parts_delta": -15,
        "health_delta": -10,
        "morale_delta": -10,
        "sickness": False,
    },
    {
        "name": "Singing Around the Campfire",
        "description": "The group spends the evening singing and swapping stories.",
        "food_delta": 0,
        "parts_delta": 0,
        "health_delta": 3,
        "morale_delta": 15,
        "sickness": False,
    },
    {
        "name": "Wheel Breaks",
        "description": "A wagon wheel cracks on a rocky outcrop.",
        "food_delta": 0,
        "parts_delta": -15,
        "health_delta": 0,
        "morale_delta": -4,
        "sickness": False,
    },
    {
        "name": "Cholera Scare",
        "description": "The party passes near contaminated water; several become ill.",
        "food_delta": 0,
        "parts_delta": 0,
        "health_delta": -20,
        "morale_delta": -12,
        "sickness": True,
    },
]


class EventSystem:
    """Generates and applies random events each day."""

    # Probability that any event occurs on a given day.
    # Lower than before to reduce relentless stacking penalties.
    BASE_EVENT_CHANCE = 0.22

    def __init__(self) -> None:
        # Keep a tiny bit of memory so event selection can avoid
        # repetitive severe punishment streaks.
        self._last_event_severity: str = "neutral"

    def _event_severity(self, event_data: dict) -> str:
        """Classify an event into a rough severity tier.

        This is intentionally simple and transparent, so balancing is easy.
        """
        total_delta = (
            event_data.get("food_delta", 0)
            + event_data.get("parts_delta", 0)
            + event_data.get("health_delta", 0)
            + event_data.get("morale_delta", 0)
            + event_data.get("miles_delta", 0)
        )
        if total_delta <= -25 or event_data.get("sickness"):
            return "severe_negative"
        if total_delta < 0:
            return "mild_negative"
        if total_delta > 10:
            return "positive"
        return "neutral"

    def _event_weight(self, severity: str) -> float:
        """Return sampling weight by severity category."""
        weights = {
            "neutral": 1.2,
            "mild_negative": 1.0,
            "severe_negative": 0.35,
            "positive": 0.9,
        }
        return weights[severity]

    def _choose_event(self) -> dict:
        """Pick an event using weighted severity and anti-streak logic."""
        weighted_events = []
        weighted_values = []
        for event in EVENT_CATALOGUE:
            severity = self._event_severity(event)
            # Anti-streak protection: if yesterday was severe negative,
            # strongly suppress severe negatives today.
            anti_streak_factor = 0.2 if (
                self._last_event_severity == "severe_negative"
                and severity == "severe_negative"
            ) else 1.0
            weighted_events.append(event)
            weighted_values.append(self._event_weight(severity) * anti_streak_factor)

        chosen = random.choices(weighted_events, weights=weighted_values, k=1)[0]
        self._last_event_severity = self._event_severity(chosen)
        return chosen

    def roll(self, world: "WagonTrain", agents: List["Agent"]) -> List[str]:
        """Potentially trigger a random event; return list of messages."""
        messages: List[str] = []

        if random.random() > self.BASE_EVENT_CHANCE:
            return messages  # No event today

        event_data = self._choose_event()
        messages.append(f"[EVENT] {event_data['name']}: {event_data['description']}")

        living = [a for a in agents if a.alive]

        # Apply food / parts / morale deltas
        if event_data.get("food_delta", 0):
            world.food_supply = max(0.0, world.food_supply + event_data["food_delta"])
            sign = "+" if event_data["food_delta"] > 0 else ""
            messages.append(f"  Food supply: {sign}{event_data['food_delta']:.0f}")

        if event_data.get("parts_delta", 0):
            world.wagon_parts = max(0.0, min(100.0, world.wagon_parts + event_data["parts_delta"]))
            sign = "+" if event_data["parts_delta"] > 0 else ""
            messages.append(f"  Wagon parts: {sign}{event_data['parts_delta']:.0f}")

        if event_data.get("morale_delta", 0):
            world.morale = max(0.0, min(100.0, world.morale + event_data["morale_delta"]))

        if event_data.get("miles_delta", 0):
            world.miles_traveled = max(0.0, world.miles_traveled + event_data["miles_delta"])

        # Apply health delta
        health_delta = event_data.get("health_delta", 0)
        if health_delta and living:
            target = event_data.get("target")
            if target == "random":
                victim = random.choice(living)
                victim.health -= abs(health_delta)
                messages.append(
                    f"  {victim.name} is seriously affected! Health -{abs(health_delta):.0f}"
                )
            else:
                for agent in living:
                    agent.health += health_delta
                if health_delta < 0:
                    messages.append(f"  All travelers lose {abs(health_delta):.0f} health.")
                else:
                    messages.append(f"  All travelers gain {health_delta:.0f} health.")

        # Record sickness
        if event_data.get("sickness"):
            world.sickness_events.append(event_data["name"])

        return messages
