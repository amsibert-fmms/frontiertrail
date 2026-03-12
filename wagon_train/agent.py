"""Agent class for the wagon train simulator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Role(str, Enum):
    CAPTAIN = "captain"
    HUNTER = "hunter"
    MEDIC = "medic"
    WHEELWRIGHT = "wheelwright"
    SCOUT = "scout"
    PASSENGER = "passenger"
    GUARD = "guard"
    BLACKSMITH = "blacksmith"
    HOSTLER = "hostler"
    COOK = "cook"


# Base influence scores by role
_ROLE_BASE_INFLUENCE: dict[Role, float] = {
    Role.CAPTAIN: 3.0,
    Role.HUNTER: 2.0,
    Role.MEDIC: 2.0,
    Role.WHEELWRIGHT: 2.0,
    Role.SCOUT: 1.5,
    Role.PASSENGER: 1.0,
    Role.GUARD: 1.6,
    Role.BLACKSMITH: 1.7,
    Role.HOSTLER: 1.5,
    Role.COOK: 1.4,
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
        # ELI5: every traveler starts the trip with a route destination in mind.
        # This preference is a starting anchor, not a permanent lock.
        self.intended_route: str = self._initial_trail_intent()

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
        """Return baseline dynamic influence without world-context boosts.

        ELI5:
        - This is the old/simple influence model kept for compatibility.
        - It only looks at this person's current health and morale.
        - Vote collection can optionally call `effective_influence_for_world`
          when it wants role-context boosts ("needed right now" behavior).
        """
        if not self.alive:
            return 0.0
        health_factor = self._health / 100.0
        morale_factor = self._morale / 100.0
        return self.influence * (0.5 * health_factor + 0.5 * morale_factor)

    def effective_influence_for_world(self, world: "WagonTrain") -> float:  # noqa: F821
        """Return influence with moderate context-sensitive role boosts.

        ELI5:
        - People should be heard a bit more when their specialty is urgently
          needed (doctor when sick, mechanic when wagon is failing, etc.).
        - We keep boosts moderate so no role can fully dominate voting.
        """
        base = self.effective_influence
        if base <= 0.0:
            return 0.0

        boost_multiplier = 1.0

        # Hunter service is "needed" when food is low.
        # Moderate boost: +20%.
        if self.role == Role.HUNTER and world.food_supply < 30.0:
            boost_multiplier += 0.20

        # Wheelwright service is "needed" when parts are low or an urgent repair
        # assignment exists in the world.
        # Moderate boost: +25%.
        if self.role == Role.WHEELWRIGHT and (
            world.wagon_parts < 40.0 or world.urgent_repair_assignee is not None
        ):
            boost_multiplier += 0.25

        # Medic service is "needed" when sickness is active.
        # Moderate boost: +25%.
        if self.role == Role.MEDIC and world.sickness_count > 0:
            boost_multiplier += 0.25

        # Captain service is "needed" when morale is low.
        # Moderate boost: +20%.
        if self.role == Role.CAPTAIN and world.morale < 60.0:
            boost_multiplier += 0.20

        # Cook service matters more when food pressure is high.
        if self.role == Role.COOK and world.food_days_remaining < 6.0:
            boost_multiplier += 0.15

        return base * boost_multiplier

    def _initial_trail_intent(self) -> str:
        """Set this traveler's starting route intent at journey start.

        ELI5:
        - Adventure-heavy roles lean California.
        - Care and stability roles lean Oregon.
        - Personal risk tolerance nudges either way.
        """
        intent_score = 0.0

        if self.role in (Role.HUNTER, Role.SCOUT, Role.GUARD):
            intent_score += 0.35
        if self.role in (Role.MEDIC, Role.COOK, Role.HOSTLER):
            intent_score -= 0.35

        intent_score += (self.traits.risk_tolerance - 0.5) * 0.8
        intent_score -= (self.traits.cooperation - 0.5) * 0.2

        return "california" if intent_score >= 0.0 else "oregon"

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


    def preferred_trail(self, world: "WagonTrain") -> str:  # noqa: F821
        """Return preferred route, allowing hardship-driven mind changes.

        ELI5:
        - Travelers begin with `intended_route`.
        - If conditions become exceptionally bad, they may switch priorities.
        - Hunger pushes toward the shorter route (California in this model).
        - Sickness/exhaustion pushes toward the easier route (Oregon in this model).
        """
        preferred = self.intended_route

        # Build route scores from the traveler's starting intent.
        route_scores = {
            "oregon": 0.0,
            "california": 0.0,
        }
        route_scores[preferred] += 1.0

        # Role biases are a soft nudge, not a hard lock.
        if self.role in (Role.HUNTER, Role.SCOUT, Role.GUARD):
            route_scores["california"] += 0.25
        if self.role in (Role.MEDIC, Role.COOK, Role.HOSTLER):
            route_scores["oregon"] += 0.25

        # Distress signals that can override starting intent.
        exceptionally_hungry = self.hunger >= 75.0 or world.food_days_remaining <= 2.5
        exceptionally_sick = self.health <= 35.0 or world.sickness_count >= 2
        exceptionally_tired = self.morale <= 30.0 or world.morale <= 40.0

        # ELI5: after the 1849 Gold Rush, California became more attractive,
        # so we add a gentle historical nudge in those years.
        if world.start_date.year >= 1849:
            route_scores["california"] += 0.3

        # Hungry travelers are willing to gamble for what they perceive as shorter.
        if exceptionally_hungry:
            route_scores["california"] += 2.0
            # ELI5: truly low food-days creates panic for perceived short-path gains.
            if world.food_days_remaining <= 1.5:
                route_scores["california"] += 0.5

        # Sick/tired travelers prefer easier and more conservative routing.
        if exceptionally_sick:
            route_scores["oregon"] += 1.0
        if exceptionally_tired:
            route_scores["oregon"] += 0.9

        # Stubbornness resists changing from original intent.
        route_scores[self.intended_route] += self.traits.stubbornness * 0.8

        return max(route_scores, key=lambda route: route_scores[route])

    def propose_trail_plan(self, world: "WagonTrain") -> str:  # noqa: F821
        """Propose 'oregon', 'california', or 'split' at Soda Springs.

        ELI5:
        - Most people cast a direct route vote.
        - Split votes are now rarer and mostly appear when conditions are stable.
        - Under high distress, people avoid splitting and pick a direct route.
        """
        high_distress = (
            world.food_days_remaining <= 4.0
            or world.sickness_count >= 1
            or self.morale <= 45.0
            or self.health <= 45.0
        )

        if (
            not high_distress
            and self.traits.cooperation >= 0.82
            and self.traits.stubbornness <= 0.28
        ):
            return "split"
        return self.preferred_trail(world)

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
            # ELI5: hunters are our best chance to refill food quickly.
            # If we have fewer than ~5 days of food left, prioritize hunting.
            return Action.HUNT
        # If this traveler has been explicitly assigned an urgent wagon repair,
        # they should immediately advocate for repair until the team handles it.
        if world.urgent_repair_assignee == self.name:
            return Action.REPAIR_WAGON

        # Role-specific tendencies
        if self.role == Role.HUNTER and world.food_supply < 30:
            return Action.HUNT

        if self.role in (Role.WHEELWRIGHT, Role.BLACKSMITH) and world.wagon_parts < 30:
            return Action.REPAIR_WAGON

        if self.role == Role.MEDIC and world.sickness_count > 0:
            return Action.REST

        if self.role == Role.SCOUT:
            # Scouts prefer travelling; choose crossing method based on risk tolerance
            if world.river_ahead:
                return self._scout_river_decision(world)
            return Action.TRAVEL

        if self.role == Role.CAPTAIN:
            return self._captain_decision(world)

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

    def _captain_decision(self, world: "WagonTrain") -> "Action":  # noqa: F821
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

    def _critical_survival_state(self, world: "WagonTrain") -> bool:  # noqa: F821
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

        # Landmark urgency context:
        # ELI5: when a fort is very close, we should push hard to reach it,
        # because it can resupply and stabilize the run.
        distance_to_next_landmark = world.distance_to_next_landmark
        next_landmark_is_fort = world.is_fort_stop(world.next_landmark)

        # Each low-progress day should make movement actions more appealing.
        # ELI5: this is a "we are getting stuck" alarm that gets louder each day.
        stall_pressure = min(9.0, world.low_progress_streak * 1.35)

        # Stagnation panic boost:
        # ELI5: after 3+ low-progress days in a row, add a strong extra push
        # toward travel so we break out of repeated maintenance/hunt loops.
        stagnation_travel_boost = 2.0 if world.low_progress_streak >= 3 else 0.0

        # Non-travel pressure applies only when behind schedule.
        # ELI5: if we are late, we put a small "cost" on actions that do not move us.
        # This is intentionally capped so survival actions can still win when truly needed.
        non_travel_pressure = min(3.0, max(0.0, behind_schedule_ratio - 1.0) * 2.0 + world.low_progress_streak * 0.18)

        # Fort-proximity urgency bias for travel:
        # - medium bump under 120 miles
        # - larger bump under 80 miles
        # - strongest bump under 40 miles
        # These stack so the urgency rises quickly as a fort gets close.
        fort_travel_urgency = 0.0
        if next_landmark_is_fort:
            if distance_to_next_landmark < 120.0:
                fort_travel_urgency += 1.5
            if distance_to_next_landmark < 80.0:
                fort_travel_urgency += 2.5
            if distance_to_next_landmark < 40.0:
                fort_travel_urgency += 4.0

        # Lightly reduce non-travel action appeal near a fort because
        # supplies/repairs are coming soon if we just keep moving.
        fort_non_travel_penalty = 0.5 if next_landmark_is_fort else 0.0

        score_map = {
            # Travel and ford are the two direct progress actions, so they absorb
            # the strongest boost from schedule pressure + stalling pressure.
            Action.TRAVEL: 5.2 + min(8.8, progress_urgency * 0.9) + stall_pressure + stagnation_travel_boost + fort_travel_urgency + (1.0 if next_landmark_is_fort else 0.0),
            Action.FORD_RIVER: 4.2 + (2.2 if world.river_ahead else -3.0) + stall_pressure,

            # Non-progress actions keep their survival/resource benefits,
            # then receive a small late-schedule penalty.
            Action.REST: (2.2 + (4.0 if self._health < 50 else 0.0)) - non_travel_pressure,
            Action.HUNT: (2.6 + (4.6 if world.food_supply < 25 else 0.0)) - non_travel_pressure - fort_non_travel_penalty,
            Action.REPAIR_WAGON: (2.5 + (4.0 if world.wagon_parts < 30 else 0.0)) - non_travel_pressure - fort_non_travel_penalty,
            Action.RATION_FOOD: (1.6 + (3.7 if world.food_supply < 20 else 0.0)) - non_travel_pressure,
            # Waiting is usually a low-progress move; it is only attractive
            # when a river blocks us and risk tolerance is low.
            Action.WAIT_AT_RIVER: (0.6 + (2.2 if world.river_ahead else -2.5)) - non_travel_pressure,
            # Ferry is a moderate-risk progress option at rivers, but it costs food.
            Action.FERRY_ACROSS: (2.4 + (2.0 if world.river_ahead else -2.0) - (1.2 if world.food_supply < 15 else 0.0)),
            # Caulking is another river option with slight parts wear pressure.
            Action.CAULK_WAGON: (2.2 + (2.0 if world.river_ahead else -2.0) - (0.6 if world.wagon_parts < 20 else 0.0)),
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
