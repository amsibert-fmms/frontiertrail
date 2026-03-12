"""Action enum and decision engine for the wagon train simulator."""

from __future__ import annotations

import random
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .agent import Agent
    from .world import WagonTrain


class Action(str, Enum):
    TRAVEL = "travel"
    REST = "rest"
    HUNT = "hunt"
    REPAIR_WAGON = "repair_wagon"
    RATION_FOOD = "ration_food"
    FORD_RIVER = "ford_river"


# ------------------------------------------------------------------
# Action outcome constants
# ------------------------------------------------------------------

# Miles gained per travel day (base)
# We intentionally increased this baseline so that, after accounting for
# non-travel days, weather, and setbacks, successful runs are actually feasible.
TRAVEL_BASE_MILES = 32.0

# Food consumed per person per day baseline
FOOD_PER_PERSON_PER_DAY = 1.5

# Food gained from a successful hunt (base range)
HUNT_FOOD_MIN = 30.0
HUNT_FOOD_MAX = 70.0
# If hunting fails, the party can still forage a small amount.
HUNT_FOOD_FAIL_MIN = 5.0
HUNT_FOOD_FAIL_MAX = 12.0

# Parts restored by a repair action
REPAIR_PARTS_RESTORE = 20.0

# Ration food reduces consumption but costs morale
RATION_CONSUMPTION_FACTOR = 0.4   # 40 % of normal consumption

# Health cost of fording a river (base range per person)
FORD_HEALTH_COST_MIN = 4.0
FORD_HEALTH_COST_MAX = 15.0

# Miles gained when fording
FORD_MILES = 12.0


class DecisionEngine:
    """Resolves group decisions via weighted voting by agent influence."""

    def collect_votes(
        self, agents: List["Agent"], world: "WagonTrain"
    ) -> Dict[Action, float]:
        """Return a mapping of action → total influence weight."""
        tally: Dict[Action, float] = Counter()
        for agent in agents:
            if not agent.alive:
                continue
            proposed = agent.propose_action(world)
            tally[proposed] += agent.effective_influence
        return dict(tally)

    def resolve(
        self, agents: List["Agent"], world: "WagonTrain"
    ) -> Tuple[Action, Dict[Action, float]]:
        """Choose the winning action and return it with the vote tally."""
        tally = self.collect_votes(agents, world)
        if not tally:
            return Action.REST, tally
        winning_action = max(tally, key=lambda a: tally[a])
        return winning_action, tally

    # ------------------------------------------------------------------
    # Apply action to world state
    # ------------------------------------------------------------------

    def apply_action(
        self,
        action: Action,
        agents: List["Agent"],
        world: "WagonTrain",
    ) -> List[str]:
        """Apply the chosen group action and return a list of outcome messages."""
        outcomes: List[str] = []
        living = [a for a in agents if a.alive]
        n = len(living)

        if action == Action.TRAVEL:
            outcomes.extend(self._do_travel(living, world, n))

        elif action == Action.REST:
            outcomes.extend(self._do_rest(living, world, n))

        elif action == Action.HUNT:
            outcomes.extend(self._do_hunt(living, world, n))

        elif action == Action.REPAIR_WAGON:
            outcomes.extend(self._do_repair(living, world, n))

        elif action == Action.RATION_FOOD:
            outcomes.extend(self._do_ration(living, world, n))

        elif action == Action.FORD_RIVER:
            outcomes.extend(self._do_ford(living, world, n))

        # Always consume some food (modified by action)
        outcomes.extend(self._consume_food(living, world, action))

        # Update group morale based on world state
        self._update_morale(world, living)

        return outcomes

    # ------------------------------------------------------------------
    # Individual action handlers
    # ------------------------------------------------------------------

    def _do_travel(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        if world.river_ahead:
            msgs.append(
                "The group attempts to travel but a river blocks the path — "
                "they must ford or wait."
            )
            return msgs

        modifier = world.travel_modifier
        # Wagon-parts factor:
        # - Old model punished low parts very harshly.
        # - New model keeps a floor so the party is slowed, not frozen.
        # - We also cap top-end bonus to keep things predictable.
        parts_factor = min(1.2, 0.5 + world.wagon_parts / 100.0)
        miles = TRAVEL_BASE_MILES * modifier * parts_factor
        miles *= random.uniform(0.8, 1.2)  # ±20% random variation
        world.miles_traveled += miles
        msgs.append(f"The wagon train travels {miles:.1f} miles ({world.weather.value} weather).")

        # Fatigue: slight health / morale cost
        for agent in living:
            agent.health -= random.uniform(0.5, 2.0)
            agent.hunger += random.uniform(1.0, 3.0)
        return msgs

    def _do_rest(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        msgs.append("The party rests for the day, recovering strength.")
        for agent in living:
            agent.health = min(100.0, agent.health + random.uniform(3.0, 8.0))
            agent.morale = min(100.0, agent.morale + random.uniform(2.0, 5.0))
        world.morale = min(100.0, world.morale + 3.0)
        return msgs

    def _do_hunt(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        # Find the best hunter(s)
        from .agent import Role
        hunters = [a for a in living if a.role == Role.HUNTER]
        # Hunting success model:
        # - still rewards having hunters,
        # - but no near-guaranteed outcomes with a small number of specialists.
        skill = 0.3 + len(hunters) * 0.2
        success_chance = min(0.9, skill)

        # Weather affects hunt reliability and yields.
        # Bad weather means less activity/visibility and harder tracking.
        weather_hunt_factor = {
            "sunny": 1.0,
            "cloudy": 0.95,
            "hot": 0.9,
            "rainy": 0.8,
            "snowy": 0.75,
            "stormy": 0.6,
        }[world.weather.value]
        success_chance *= weather_hunt_factor

        if random.random() < success_chance:
            food_gained = random.uniform(HUNT_FOOD_MIN, HUNT_FOOD_MAX) * weather_hunt_factor
            world.food_supply += food_gained
            msgs.append(f"The hunt is successful! +{food_gained:.1f} lbs of food.")
            for agent in living:
                agent.morale = min(100.0, agent.morale + 3.0)
        else:
            # Even on a "failed" hunt, basic foraging usually finds something.
            consolation_food = random.uniform(HUNT_FOOD_FAIL_MIN, HUNT_FOOD_FAIL_MAX)
            world.food_supply += consolation_food
            msgs.append(
                "The hunting party has little luck, "
                f"but foraging adds +{consolation_food:.1f} lbs of food."
            )
            for agent in living:
                agent.morale = max(0.0, agent.morale - 2.0)
        return msgs

    def _do_repair(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        from .agent import Role
        mechanics = [a for a in living if a.role == Role.MECHANIC]
        bonus = len(mechanics) * 5.0
        restored = REPAIR_PARTS_RESTORE + bonus
        world.wagon_parts = min(100.0, world.wagon_parts + restored)
        # Once repair action is completed, clear any urgent person-specific repair duty.
        world.urgent_repair_assignee = None
        msgs.append(
            f"The wagon is repaired. Parts supply restored by {restored:.0f} points "
            f"(now {world.wagon_parts:.0f})."
        )
        return msgs

    def _do_ration(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        msgs.append(
            "Food is rationed strictly. Everyone receives reduced portions today."
        )
        for agent in living:
            agent.morale = max(0.0, agent.morale - 3.0)
            agent.hunger = min(100.0, agent.hunger + 5.0)
        world.morale = max(0.0, world.morale - 2.0)
        return msgs

    def _do_ford(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        if not world.river_ahead:
            msgs.append("There is no river to ford — the party presses on.")
            world.miles_traveled += 5.0
            return msgs

        msgs.append("The party attempts to ford the river!")
        # Risk varies; scouts reduce danger
        from .agent import Role
        scouts = [a for a in living if a.role == Role.SCOUT]
        risk = max(0.1, 0.6 - len(scouts) * 0.1)

        if random.random() < risk:
            # Bad crossing
            health_loss = random.uniform(FORD_HEALTH_COST_MIN, FORD_HEALTH_COST_MAX)
            for agent in living:
                agent.health -= random.uniform(0, health_loss)
            food_lost = random.uniform(5.0, 20.0)
            world.food_supply = max(0.0, world.food_supply - food_lost)
            world.morale = max(0.0, world.morale - 10.0)
            msgs.append(
                f"The crossing was treacherous! Everyone lost health; "
                f"{food_lost:.1f} lbs of food were lost."
            )
        else:
            msgs.append("The party fords the river safely!")
            for agent in living:
                agent.morale = min(100.0, agent.morale + 5.0)
            # Safe crossing can also improve confidence and preserve equipment.
            world.morale = min(100.0, world.morale + 3.0)
            world.wagon_parts = min(100.0, world.wagon_parts + 2.0)

        world.miles_traveled += FORD_MILES
        world.river_crossed()
        return msgs

    # ------------------------------------------------------------------
    # Food consumption
    # ------------------------------------------------------------------

    def _consume_food(
        self,
        living: List["Agent"],
        world: "WagonTrain",
        action: Action,
    ) -> List[str]:
        msgs: List[str] = []
        factor = RATION_CONSUMPTION_FACTOR if action == Action.RATION_FOOD else 1.0
        consumed = len(living) * FOOD_PER_PERSON_PER_DAY * factor
        world.food_supply = max(0.0, world.food_supply - consumed)

        if world.food_supply <= 0:
            msgs.append("The food supply is exhausted! Everyone is starving.")
            for agent in living:
                agent.hunger = min(100.0, agent.hunger + 15.0)
                agent.health -= 10.0
        else:
            for agent in living:
                agent.hunger = max(0.0, agent.hunger - FOOD_PER_PERSON_PER_DAY * factor * 5)
        return msgs

    # ------------------------------------------------------------------
    # Morale update
    # ------------------------------------------------------------------

    def _update_morale(
        self, world: "WagonTrain", living: List["Agent"]
    ) -> None:
        avg_health = (
            sum(a.health for a in living) / len(living) if living else 0.0
        )
        avg_morale = (
            sum(a.morale for a in living) / len(living) if living else 0.0
        )
        # Drift group morale toward average individual morale
        world.morale += (avg_morale - world.morale) * 0.1

        # Additional world-pressure penalties (stronger than before):
        # - repeated low/no progress hurts confidence,
        # - hunger and low health increase hopelessness,
        # - broken equipment + bad weather reduce spirit.
        weather_penalty_map = {
            "sunny": 0.0,
            "cloudy": 0.4,
            "hot": 0.8,
            "rainy": 1.6,
            "snowy": 2.2,
            "stormy": 3.0,
        }
        avg_hunger = sum(a.hunger for a in living) / len(living) if living else 0.0
        progress_penalty = min(6.0, world.low_progress_streak * 0.8)
        hunger_penalty = max(0.0, (avg_hunger - 40.0) * 0.05)
        health_penalty = max(0.0, (55.0 - avg_health) * 0.07)
        mechanical_penalty = max(0.0, (45.0 - world.wagon_parts) * 0.05)
        weather_penalty = weather_penalty_map[world.weather.value]

        world.morale -= (
            progress_penalty
            + hunger_penalty
            + health_penalty
            + mechanical_penalty
            + weather_penalty
        )
        world.morale = max(0.0, min(100.0, world.morale))

        # Sync individual morale toward group morale slightly
        for agent in living:
            agent.morale += (world.morale - agent.morale) * 0.05
