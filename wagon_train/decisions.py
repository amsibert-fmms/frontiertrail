"""Action enum and decision engine for the wagon train simulator."""

from __future__ import annotations

import random
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple

from .world import FOOD_PER_PERSON_PER_DAY

if TYPE_CHECKING:
    from .agent import Agent
    from .world import WagonTrain


def _adjust_relationship(agent: "Agent", other_name: str, delta: float) -> None:
    """Clamp-safe adjustment of agent's trust toward another agent by name."""
    current = agent.relationships.get(other_name, 0.5)
    agent.relationships[other_name] = max(0.0, min(1.0, current + delta))


class Action(str, Enum):
    TRAVEL = "travel"
    REST = "rest"
    HUNT = "hunt"
    REPAIR_WAGON = "repair_wagon"
    RATION_FOOD = "ration_food"
    FORD_RIVER = "ford_river"
    WAIT_AT_RIVER = "wait_at_river"
    CAULK_WAGON = "caulk_wagon"
    FERRY_ACROSS = "ferry_across"


# ------------------------------------------------------------------
# Action outcome constants
# ------------------------------------------------------------------

# Miles gained per travel day (base)
TRAVEL_BASE_MILES = 20.0

# Food gained from a successful hunt (base range)
HUNT_FOOD_MIN = 30.0
HUNT_FOOD_MAX = 70.0

# Parts restored by a repair action
REPAIR_PARTS_RESTORE = 20.0

# Ration food reduces consumption but costs morale
RATION_CONSUMPTION_FACTOR = 0.4   # 40 % of normal consumption

# Health cost of fording a river (base range per person)
FORD_HEALTH_COST_MIN = 5.0
FORD_HEALTH_COST_MAX = 25.0

# Miles gained when fording / crossing a river
FORD_MILES = 5.0

# Food cost of taking a ferry (simulates trading supplies)
FERRY_COST_MIN = 10.0
FERRY_COST_MAX = 20.0


class DecisionEngine:
    """Resolves group decisions via weighted voting by agent influence."""

    def collect_votes(
        self, agents: List["Agent"], world: "WagonTrain"
    ) -> Dict[Action, float]:
        """Return a mapping of action → total influence weight.

        Each agent's effective influence is scaled by their relationship
        modifier — a measure of how much trust they have earned from peers.
        """
        living = [a for a in agents if a.alive]
        tally: Dict[Action, float] = Counter()
        for agent in agents:
            if not agent.alive:
                continue
            proposed = agent.propose_action(world)
            modifier = agent.get_relationship_modifier(living)
            tally[proposed] += agent.effective_influence * modifier
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

        elif action == Action.WAIT_AT_RIVER:
            outcomes.extend(self._do_wait_river(living, world, n))

        elif action == Action.CAULK_WAGON:
            outcomes.extend(self._do_caulk(living, world, n))

        elif action == Action.FERRY_ACROSS:
            outcomes.extend(self._do_ferry(living, world, n))

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
        # Wagon-parts penalty
        parts_factor = min(1.0, world.wagon_parts / 50.0)
        miles = TRAVEL_BASE_MILES * modifier * parts_factor
        miles *= random.uniform(0.8, 1.2)  # ±20% random variation
        world.miles_traveled += miles
        # Track speed for derived metric (rolling 7-day window)
        world._recent_miles.append(miles)
        if len(world._recent_miles) > 7:
            world._recent_miles.pop(0)
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
        skill = len(hunters) * 0.3 + 0.4  # baseline success chance
        success_chance = min(0.95, skill)

        if random.random() < success_chance:
            food_gained = random.uniform(HUNT_FOOD_MIN, HUNT_FOOD_MAX)
            world.food_supply += food_gained
            msgs.append(f"The hunt is successful! +{food_gained:.1f} lbs of food.")
            for agent in living:
                agent.morale = min(100.0, agent.morale + 3.0)
            # Hunters earn trust from the party
            for agent in living:
                for hunter in hunters:
                    if agent is not hunter:
                        _adjust_relationship(agent, hunter.name, +0.05)
        else:
            msgs.append("The hunting party returns empty-handed.")
            for agent in living:
                agent.morale = max(0.0, agent.morale - 2.0)
            # Slight loss of trust in hunters
            for agent in living:
                for hunter in hunters:
                    if agent is not hunter:
                        _adjust_relationship(agent, hunter.name, -0.03)
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
            # Scouts lose trust after a dangerous ford
            for agent in living:
                for scout in scouts:
                    if agent is not scout:
                        _adjust_relationship(agent, scout.name, -0.05)
        else:
            msgs.append("The party fords the river safely!")
            for agent in living:
                agent.morale = min(100.0, agent.morale + 5.0)
            # Scouts earn trust for a safe crossing
            for agent in living:
                for scout in scouts:
                    if agent is not scout:
                        _adjust_relationship(agent, scout.name, +0.05)

        world.miles_traveled += FORD_MILES
        world.river_crossed()
        return msgs

    # ------------------------------------------------------------------
    # New river crossing methods
    # ------------------------------------------------------------------

    def _do_wait_river(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        """Wait at the riverbank — scouts conditions and rests the party."""
        msgs: List[str] = []
        if not world.river_ahead:
            return self._do_rest(living, world, n)
        msgs.append(
            "The party waits at the riverbank, scouting for a safer crossing opportunity."
        )
        for agent in living:
            agent.health = min(100.0, agent.health + random.uniform(2.0, 5.0))
            agent.morale = min(100.0, agent.morale + random.uniform(1.0, 3.0))
        world.morale = min(100.0, world.morale + 1.0)
        return msgs

    def _do_caulk(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        """Caulk the wagon and float it across — moderate risk, no food cost."""
        msgs: List[str] = []
        if not world.river_ahead:
            msgs.append("There is no river — the wagon continues.")
            world.miles_traveled += FORD_MILES
            return msgs

        msgs.append("The party caulks the wagon and attempts to float across!")
        from .agent import Role
        scouts = [a for a in living if a.role == Role.SCOUT]
        risk = max(0.1, 0.3 - len(scouts) * 0.05)

        if random.random() < risk:
            # Wagon takes on water — parts damaged
            parts_lost = random.uniform(10.0, 25.0)
            world.wagon_parts = max(0.0, world.wagon_parts - parts_lost)
            world.morale = max(0.0, world.morale - 8.0)
            for agent in living:
                agent.health -= random.uniform(0.0, 10.0)
            msgs.append(
                f"The wagon nearly capsized! Parts damaged ({parts_lost:.1f} lost)."
            )
            for agent in living:
                for scout in scouts:
                    if agent is not scout:
                        _adjust_relationship(agent, scout.name, -0.05)
        else:
            msgs.append("The wagon floats across without incident!")
            for agent in living:
                agent.morale = min(100.0, agent.morale + 3.0)
            for agent in living:
                for scout in scouts:
                    if agent is not scout:
                        _adjust_relationship(agent, scout.name, +0.05)

        world.miles_traveled += FORD_MILES
        world.river_crossed()
        return msgs

    def _do_ferry(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        """Pay for ferry passage — safe crossing but costs food supplies."""
        msgs: List[str] = []
        if not world.river_ahead:
            msgs.append("There is no river — the wagon continues.")
            world.miles_traveled += FORD_MILES
            return msgs

        ferry_cost = random.uniform(FERRY_COST_MIN, FERRY_COST_MAX)
        if world.food_supply < ferry_cost:
            msgs.append(
                "The party cannot afford ferry passage — attempting to ford instead!"
            )
            return self._do_ford(living, world, n)

        world.food_supply = max(0.0, world.food_supply - ferry_cost)
        msgs.append(
            f"The party trades {ferry_cost:.1f} lbs of supplies for ferry passage."
        )
        msgs.append("The ferry carries everyone safely across the river.")
        for agent in living:
            agent.morale = min(100.0, agent.morale + 5.0)

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
        # Store derived metric on world for agent reasoning
        world.avg_health = avg_health

        # Drift group morale toward average individual morale
        world.morale += (avg_morale - world.morale) * 0.1
        world.morale = max(0.0, min(100.0, world.morale))

        # Sync individual morale toward group morale slightly
        for agent in living:
            agent.morale += (world.morale - agent.morale) * 0.05
