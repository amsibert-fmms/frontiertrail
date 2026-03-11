"""Agent class for the wagon train simulator."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Role(str, Enum):
    LEADER = "leader"
    HUNTER = "hunter"
    MEDIC = "medic"
    MECHANIC = "mechanic"
    SCOUT = "scout"
    PASSENGER = "passenger"


# Base influence scores by role
_ROLE_BASE_INFLUENCE: dict[Role, float] = {
    Role.LEADER: 3.0,
    Role.HUNTER: 2.0,
    Role.MEDIC: 2.0,
    Role.MECHANIC: 2.0,
    Role.SCOUT: 1.5,
    Role.PASSENGER: 1.0,
}


@dataclass
class Traits:
    """Personality traits that influence an agent's decisions.

    All values are floats in the range [0.0, 1.0].
    """

    risk_tolerance: float = 0.5   # 0=cautious, 1=reckless
    generosity: float = 0.5       # 0=selfish, 1=generous
    stubbornness: float = 0.5     # 0=flexible, 1=stubborn
    cooperation: float = 0.5      # 0=independent, 1=cooperative

    def __post_init__(self) -> None:
        for attr in ("risk_tolerance", "generosity", "stubbornness", "cooperation"):
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Trait '{attr}' must be in [0.0, 1.0], got {value}")

    @classmethod
    def random(cls) -> "Traits":
        return cls(
            risk_tolerance=round(random.uniform(0.1, 0.9), 2),
            generosity=round(random.uniform(0.1, 0.9), 2),
            stubbornness=round(random.uniform(0.1, 0.9), 2),
            cooperation=round(random.uniform(0.1, 0.9), 2),
        )


class Agent:
    """Represents a single traveler in the wagon party."""

    def __init__(
        self,
        name: str,
        role: Role,
        traits: Optional[Traits] = None,
        health: float = 100.0,
        hunger: float = 0.0,
        morale: float = 80.0,
        influence: Optional[float] = None,
    ) -> None:
        self.name = name
        self.role = role
        self.traits: Traits = traits if traits is not None else Traits.random()
        self._health: float = float(health)
        self._hunger: float = float(hunger)
        self._morale: float = float(morale)
        self.influence: float = (
            influence if influence is not None else _ROLE_BASE_INFLUENCE[role]
        )
        self.alive: bool = True
        # Trust scores toward other agents by name: 0.0 = distrust, 0.5 = neutral, 1.0 = full trust
        self.relationships: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties with clamping
    # ------------------------------------------------------------------

    @property
    def health(self) -> float:
        return self._health

    @health.setter
    def health(self, value: float) -> None:
        self._health = max(0.0, min(100.0, value))
        if self._health <= 0.0:
            self.alive = False

    @property
    def hunger(self) -> float:
        return self._hunger

    @hunger.setter
    def hunger(self, value: float) -> None:
        self._hunger = max(0.0, min(100.0, value))

    @property
    def morale(self) -> float:
        return self._morale

    @morale.setter
    def morale(self, value: float) -> None:
        self._morale = max(0.0, min(100.0, value))

    # ------------------------------------------------------------------
    # Effective influence (dynamic: penalised by poor health / morale)
    # ------------------------------------------------------------------

    @property
    def effective_influence(self) -> float:
        if not self.alive:
            return 0.0
        health_factor = self._health / 100.0
        morale_factor = self._morale / 100.0
        return self.influence * (0.5 * health_factor + 0.5 * morale_factor)

    # ------------------------------------------------------------------
    # Relationship system
    # ------------------------------------------------------------------

    def get_relationship_modifier(self, living: List["Agent"]) -> float:
        """Return a multiplier reflecting how much trust this agent has earned.

        The modifier is the average trust score that all other living agents
        hold toward this agent, remapped from [0, 1] → [0.5, 1.5].
        A neutral starting trust (0.5) maps to a modifier of 1.0 (no change).
        """
        others = [a for a in living if a is not self and a.alive]
        if not others:
            return 1.0
        trust_scores = [a.relationships.get(self.name, 0.5) for a in others]
        avg_trust = sum(trust_scores) / len(trust_scores)
        return 0.5 + avg_trust  # [0, 1] trust → [0.5, 1.5] modifier

    def update_relationship(self, other_name: str, delta: float) -> None:
        """Adjust trust toward another agent by *delta*, clamped to [0.0, 1.0]."""
        current = self.relationships.get(other_name, 0.5)
        self.relationships[other_name] = max(0.0, min(1.0, current + delta))

    # ------------------------------------------------------------------
    # Action proposal
    # ------------------------------------------------------------------

    def propose_action(self, world: "WagonTrain") -> "Action":  # noqa: F821
        """Propose an action based on the agent's role and world state."""
        from .decisions import Action

        if not self.alive:
            return Action.REST

        # Role-specific tendencies (use food_days_remaining for smarter forecasting)
        if self.role == Role.HUNTER and world.food_days_remaining < 5:
            return Action.HUNT

        if self.role == Role.MECHANIC and world.wagon_parts < 30:
            return Action.REPAIR_WAGON

        if self.role == Role.MEDIC and world.sickness_count > 0:
            return Action.REST

        if self.role == Role.SCOUT:
            # Scouts prefer travelling; choose crossing method based on risk tolerance
            if world.river_ahead:
                return self._scout_river_decision(world)
            return Action.TRAVEL

        if self.role == Role.LEADER:
            return self._leader_decision(world)

        # Generic decision based on traits
        return self._generic_decision(world)

    def _scout_river_decision(self, world: "WagonTrain") -> "Action":  # noqa: F821
        """Scouts choose a river-crossing strategy based on risk tolerance."""
        from .decisions import Action

        rt = self.traits.risk_tolerance
        if rt >= 0.7:
            return Action.FORD_RIVER
        if rt >= 0.4:
            return Action.CAULK_WAGON
        if rt >= 0.2:
            return Action.FERRY_ACROSS
        return Action.WAIT_AT_RIVER

    def _leader_decision(self, world: "WagonTrain") -> "Action":  # noqa: F821
        from .decisions import Action

        if world.food_days_remaining < 5:
            return Action.HUNT if random.random() < 0.5 else Action.RATION_FOOD
        if world.wagon_parts < 20:
            return Action.REPAIR_WAGON
        if world.river_ahead:
            rt = self.traits.risk_tolerance
            if rt >= 0.6:
                return Action.FORD_RIVER
            if rt >= 0.4:
                return Action.CAULK_WAGON
            if rt >= 0.2 and world.food_days_remaining >= 5:
                return Action.FERRY_ACROSS
            return Action.WAIT_AT_RIVER
        return Action.TRAVEL

    def _generic_decision(self, world: "WagonTrain") -> "Action":  # noqa: F821
        from .decisions import Action

        # Starving agents push to hunt / ration based on days of food remaining
        if self._hunger > 70:
            return Action.RATION_FOOD
        if world.food_days_remaining < 3 and random.random() < 0.6:
            return Action.HUNT

        # Low health agents rest
        if self._health < 40 or world.sickness_count > 2:
            return Action.REST

        # River decision based on risk tolerance
        if world.river_ahead:
            rt = self.traits.risk_tolerance
            if rt > 0.6:
                return Action.FORD_RIVER
            if rt > 0.4:
                return Action.CAULK_WAGON
            return Action.WAIT_AT_RIVER

        # Default: travel or rest based on morale
        if self._morale < 30:
            return Action.REST
        return Action.TRAVEL

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        return (
            f"Agent({self.name!r}, role={self.role.value}, "
            f"health={self._health:.0f}, hunger={self._hunger:.0f}, "
            f"morale={self._morale:.0f}, influence={self.effective_influence:.2f}, "
            f"{status})"
        )
