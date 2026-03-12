"""Agent class for the wagon train simulator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Role(str, Enum):
    LEADER = "leader"
    HUNTER = "hunter"
    MEDIC = "medic"
    MECHANIC = "mechanic"
    SCOUT = "scout"
    PASSENGER = "passenger"
    PREACHER = "preacher"


# Base influence scores by role
_ROLE_BASE_INFLUENCE: dict[Role, float] = {
    Role.LEADER: 3.0,
    Role.HUNTER: 2.0,
    Role.MEDIC: 2.0,
    Role.MECHANIC: 2.0,
    Role.SCOUT: 1.5,
    Role.PASSENGER: 1.0,
    Role.PREACHER: 1.2,
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
        # If this traveler has been explicitly assigned an urgent wagon repair,
        # they should immediately advocate for repair until the team handles it.
        if world.urgent_repair_assignee == self.name:
            return Action.REPAIR_WAGON

        # Religious archetype behavior:
        # - This traveler strongly prefers full-group rest on Sundays.
        # - We still allow survival overrides so the AI does not make obviously
        #   catastrophic choices (for example refusing to hunt when food is gone).
        if self.role == Role.PREACHER and world.is_sunday and not self._critical_survival_state(world):
            return Action.REST

        # Role-specific tendencies
        if self.role == Role.HUNTER and world.food_supply < 30:
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
        base_action = self._generic_decision(world)
        return self._apply_progress_bias(world, base_action)

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

      # Food emergency
      if world.food_days_remaining < 5:
          return Action.HUNT if random.random() < 0.5 else Action.RATION_FOOD

      # Critical wagon damage
      if world.wagon_parts < 20:
          return Action.REPAIR_WAGON

      # River decision
      if world.river_ahead:
          rt = self.traits.risk_tolerance

          if rt >= 0.6:
              return Action.FORD_RIVER
          if rt >= 0.4:
              return Action.CAULK_WAGON
          if rt >= 0.2 and world.food_days_remaining >= 5:
              return Action.FERRY_ACROSS

          return Action.WAIT_AT_RIVER

      # Default behavior: travel, but allow pacing logic to modify it
      return self._apply_progress_bias(world, Action.TRAVEL)

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

    def _critical_survival_state(self, world: "WagonTrain") -> bool:
        """Return True when survival needs are urgent enough to override pacing.

        In simple words: if we are in immediate danger, we should stabilize first.
        """
        return (
            self._health < 35
            or self._hunger > 80
            or world.food_supply < 12
            or world.wagon_parts < 15
            or world.sickness_count > 2
        )

    def _action_score(self, world: "WagonTrain", action: "Action") -> float:  # noqa: F821
        """Score an action with a lightweight progress/survival/risk heuristic.

        Higher score is better for the current situation.
        """
        from .decisions import Action

        # "Progress urgency" tells us how many miles/day we now need.
        # If this climbs above the baseline required pace, we are behind schedule.
        progress_urgency = world.miles_needed_per_remaining_day
        behind_schedule_ratio = progress_urgency / max(0.1, world.REQUIRED_MILES_PER_DAY)

        # Each low-progress day should make movement actions more appealing.
        # ELI5: this is a "we are getting stuck" alarm that gets louder each day.
        stall_pressure = min(9.0, world.low_progress_streak * 1.35)

        # Non-travel pressure applies only when behind schedule.
        # ELI5: if we are late, we put a small "cost" on actions that do not move us.
        # This is intentionally capped so survival actions can still win when truly needed.
        non_travel_pressure = min(3.0, max(0.0, behind_schedule_ratio - 1.0) * 2.0 + world.low_progress_streak * 0.18)

        score_map = {
            # Travel and ford are the two direct progress actions, so they absorb
            # the strongest boost from schedule pressure + stalling pressure.
            Action.TRAVEL: 5.2 + min(8.8, progress_urgency * 0.9) + stall_pressure,
            Action.FORD_RIVER: 4.2 + (2.2 if world.river_ahead else -3.0) + stall_pressure,

            # Non-progress actions keep their survival/resource benefits,
            # then receive a small late-schedule penalty.
            Action.REST: (2.2 + (4.0 if self._health < 50 else 0.0)) - non_travel_pressure,
            Action.HUNT: (2.6 + (4.6 if world.food_supply < 25 else 0.0)) - non_travel_pressure,
            Action.REPAIR_WAGON: (2.5 + (4.0 if world.wagon_parts < 30 else 0.0)) - non_travel_pressure,
            Action.RATION_FOOD: (1.6 + (3.7 if world.food_supply < 20 else 0.0)) - non_travel_pressure,
        }
        return score_map[action]

    def _apply_progress_bias(self, world: "WagonTrain", base_action: "Action") -> "Action":  # noqa: F821
        """Bias choices toward travel when schedule pressure is high.

        This avoids the party repeatedly selecting maintenance actions while
        quietly running out of days to finish the journey.
        """
        from .decisions import Action

        if self._critical_survival_state(world):
            return base_action

        # If a river is ahead, "progress bias" means preferring to ford,
        # because travel itself cannot proceed until the crossing is resolved.
        # If we have stalled for several days, reduce the threshold so progress
        # bias kicks in earlier and the party breaks out of passive loops.
        # Lower threshold a bit versus pass #1 so progress-bias engages sooner.
        # ELI5: we do not wait as long to "switch into catch-up mode".
        adaptive_pressure = min(0.40, world.low_progress_streak * 0.035)
        must_travel_threshold = world.REQUIRED_MILES_PER_DAY * (1.15 - adaptive_pressure)
        behind_schedule = world.miles_needed_per_remaining_day >= must_travel_threshold
        if not behind_schedule:
            return base_action

        preferred_progress_action = Action.FORD_RIVER if world.river_ahead else Action.TRAVEL
        if self._action_score(world, preferred_progress_action) >= self._action_score(world, base_action):
            return preferred_progress_action
        return base_action

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
