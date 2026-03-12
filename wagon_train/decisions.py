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


# ELI5: simple fort store prices; passenger fares become this spending money.
# ELI5: food at forts now costs more, so cash stretches less far.
# This keeps fort stops useful without letting one early stop overfill stores.
FORT_FOOD_PRICE_PER_LB = 0.50
# ELI5: wagon parts are also pricier to preserve repair-action relevance.
FORT_PARTS_PRICE_PER_POINT = 1.90


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
# We intentionally increased this baseline so that, after accounting for
# non-travel days, weather, and setbacks, successful runs are actually feasible.
TRAVEL_BASE_MILES = 32.0

# Food gained from a successful hunt (base range)
HUNT_FOOD_MIN = 30.0
HUNT_FOOD_MAX = 70.0
# If hunting fails, the party can still forage a small amount.
HUNT_FOOD_FAIL_MIN = 4.0
HUNT_FOOD_FAIL_MAX = 10.0

# Hunt success tuning (Pass 3A):
# We keep these as named constants so balancing can iterate safely without
# rewriting core hunt logic each pass.
#
# ELI5:
# - BASE = how likely any party is to find food at all.
# - PER_HUNTER = how much each hunter helps at first.
# - EXTRA_HUNTER = how much hunters beyond the first two still help,
#   but at a smaller "diminishing returns" amount.
# - CAP = never let hunt become near-guaranteed.
HUNT_SUCCESS_BASE = 0.30
HUNT_SUCCESS_PER_HUNTER = 0.18
HUNT_SUCCESS_EXTRA_HUNTER = 0.10
HUNT_SUCCESS_DIMINISH_AFTER = 2
HUNT_SUCCESS_CAP = 0.86

# Sunday work penalty model.
# ELI5: if the group chooses to work on Sunday, they miss a chunk of weekly
# recovery. We model that as a small health/morale hit equivalent to giving up
# about half a rest day.
SUNDAY_WORK_HEALTH_LOSS_MIN = 1.5
SUNDAY_WORK_HEALTH_LOSS_MAX = 4.0
SUNDAY_WORK_MORALE_LOSS_MIN = 1.0
SUNDAY_WORK_MORALE_LOSS_MAX = 3.0

# Parts restored by a repair action
REPAIR_PARTS_RESTORE = 20.0

# Ration food reduces consumption but costs morale
RATION_CONSUMPTION_FACTOR = 0.4   # 40 % of normal consumption

# Health cost of fording a river (base range per person).
# Tuning pass #1 narrows this band so a failed crossing is still scary
# but less likely to cause an unrecoverable full-party spiral.
FORD_HEALTH_COST_MIN = 3.0
FORD_HEALTH_COST_MAX = 10.0

# River risk model constants.
# ELI5: crossing always has danger, but scouts should noticeably lower that danger.
FORD_RISK_BASE = 0.5
FORD_RISK_PER_SCOUT_REDUCTION = 0.1
FORD_RISK_FLOOR = 0.08

# Miles gained when fording / crossing a river
FORD_MILES = 5.0
# Miles gained when fording
FORD_MILES = 12.0

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

            # Use context-aware influence so role specialists get a moderate
            # voting bump when their service is urgently needed.
            # ELI5: if the wagon is breaking, mechanics should be listened to a
            # bit more; if people are sick, medics should matter more, etc.
            tally[proposed] += agent.effective_influence_for_world(world)
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


    def collect_trail_votes(
        self, agents: List["Agent"], world: "WagonTrain"
    ) -> Dict[str, float]:
        """Collect weighted votes for Oregon, California, or split plans."""
        living = [a for a in agents if a.alive]
        tally: Dict[str, float] = Counter()
        for agent in living:
            plan = agent.propose_trail_plan(world)
            modifier = agent.get_relationship_modifier(living)
            tally[plan] += agent.effective_influence * modifier
            tally[plan] += agent.effective_influence_for_world(world)
        return dict(tally)

    def resolve_trail_choice(
        self, agents: List["Agent"], world: "WagonTrain"
    ) -> Tuple[str, Dict[str, float], str]:
        """Resolve the Soda Springs route vote with split-party handling."""
        tally = self.collect_trail_votes(agents, world)
        if not tally:
            return "oregon", tally, "No votes were cast; the party stays on the Oregon Trail."

        winning_plan = max(tally, key=lambda k: tally[k])
        if winning_plan != "split":
            route = winning_plan if winning_plan in ("oregon", "california") else "oregon"
            msg = f"Soda Springs vote selects the {route.title()} Trail."
            return route, tally, msg

        # ELI5: if "split" wins, we follow whichever resulting party has more
        # influence, as requested by the product rule.
        living = [a for a in agents if a.alive]
        route_influence = {"oregon": 0.0, "california": 0.0}
        for agent in living:
            preference = agent.preferred_trail(world)
            modifier = agent.get_relationship_modifier(living)
            route_influence[preference] += agent.effective_influence * modifier

        chosen_route = max(route_influence, key=lambda route: route_influence[route])
        msg = (
            "Soda Springs vote favored a split party; "
            f"the caravan follows the {chosen_route.title()} group "
            "because it has greater influence."
        )
        return chosen_route, tally, msg

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
        # Sunday-rest rule:
        # If the party works on Sunday, we allow it (no hard lock-out), but we
        # apply a half-rest penalty to represent missed recovery time.
        if world.is_sunday and action != Action.REST:
            outcomes.extend(self._apply_sunday_work_penalty(living))

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
        from .agent import Role
        if world.river_ahead:
            msgs.append(
                "The group attempts to travel but a river blocks the path — "
                "they must ford or wait."
            )
            return msgs

        self._buy_supplies_at_fort_if_possible(world, msgs)

        modifier = world.travel_modifier
        # Wagon-parts factor:
        # - Old model punished low parts very harshly.
        # - New model keeps a floor so the party is slowed, not frozen.
        # - We also cap top-end bonus to keep things predictable.
        parts_factor = min(1.2, 0.5 + world.wagon_parts / 100.0)
        # Draft-animal power directly affects mobility.
        # ELI5: fewer pull animals means the same wagon moves slower.
        animal_factor = world.draft_power_multiplier
        # ELI5: slightly stronger captain pace bonus helps late-game consistency,
        # especially on seeds that previously stalled just short of Oregon.
        captain_bonus = 1.0 + (0.06 * sum(1 for a in living if a.role == Role.CAPTAIN))
        miles = TRAVEL_BASE_MILES * modifier * parts_factor * animal_factor * captain_bonus
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

        # Passive role contributions during travel day:
        # - Hunters gather small trail food while moving.
        # - Mechanics perform rolling maintenance while moving.
        # These are intentionally modest so travel remains meaningful without
        # replacing dedicated hunt/repair actions.
        hunters = [a for a in living if a.role == Role.HUNTER]
        wheelwrights = [a for a in living if a.role == Role.WHEELWRIGHT]
        blacksmiths = [a for a in living if a.role == Role.BLACKSMITH]
        hostlers = [a for a in living if a.role == Role.HOSTLER]
        cooks = [a for a in living if a.role == Role.COOK]

        # Each hunter contributes about one person's daily food need.
        # FOOD_PER_PERSON_PER_DAY is already the simulation's daily baseline.
        trail_food = len(hunters) * FOOD_PER_PERSON_PER_DAY
        if trail_food > 0.0:
            world.food_supply += trail_food
            msgs.append(
                f"Hunters gather trail food while moving: +{trail_food:.1f} lbs."
            )

        # Everyone can forage a little while traveling; cooks improve usable yield.
        # ELI5: everyone still forages, just a little less than before so hunting
        # remains the primary way to refill food quickly.
        general_forage = n * 0.32
        # ELI5: cooks still help convert trail finds into meals, but the bonus is
        # reduced to avoid runaway food growth in long stable runs.
        cook_bonus = len(cooks) * 0.7
        if general_forage + cook_bonus > 0.0:
            world.food_supply += general_forage + cook_bonus
            msgs.append(
                "General trail foraging adds "
                f"+{general_forage + cook_bonus:.1f} lbs of food."
            )

        # Wheelwrights and blacksmiths both contribute to repairs.
        travel_repairs = len(wheelwrights) * 2.0 + len(blacksmiths) * 1.5
        if travel_repairs > 0.0:
            world.wagon_parts = min(100.0, world.wagon_parts + travel_repairs)
            msgs.append(
                "Wheelwrights and blacksmiths handle rolling maintenance: "
                f"+{travel_repairs:.0f} parts."
            )

        if hostlers:
            world.morale = min(100.0, world.morale + len(hostlers) * 0.8)
            msgs.append("Hostlers steady the draft team, reducing stress on the trail.")

        return msgs

    def _do_rest(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        msgs.append("The party rests for the day, recovering strength.")

        # Medic-assisted healing bonus:
        # ELI5: medics make every rest day a little more effective at recovery.
        from .agent import Role
        medics = [a for a in living if a.role == Role.MEDIC]
        cooks = [a for a in living if a.role == Role.COOK]
        medic_heal_bonus = min(4.0, len(medics) * 1.0)
        # ELI5: cooks improve rest recovery, but we cap this lower so medics stay
        # the main healing specialists and rest remains a meaningful choice.
        cook_recovery_bonus = min(1.6, len(cooks) * 0.7)

        for agent in living:
            agent.health = min(
                100.0,
                agent.health + random.uniform(3.0, 8.0) + medic_heal_bonus + cook_recovery_bonus,
            )
            agent.morale = min(100.0, agent.morale + random.uniform(2.0, 5.0))
        world.morale = min(100.0, world.morale + 3.0)

        if medic_heal_bonus > 0.0:
            msgs.append(
                "Medic care improves rest recovery: "
                f"+{medic_heal_bonus:.1f} bonus healing per traveler."
            )

        if cook_recovery_bonus > 0.0:
            msgs.append(
                "Cooks stretch warm meals and broths during camp rest: "
                f"+{cook_recovery_bonus:.1f} extra healing per traveler."
            )
        return msgs

    def _do_hunt(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        # Find the best hunter(s)
        from .agent import Role
        hunters = [a for a in living if a.role == Role.HUNTER]

        # Hunting success model (Pass 3A):
        # We intentionally rebalance hunt to avoid very high reliability that can
        # dominate decision-making and trap the party in hunt-heavy loops.
        #
        # ELI5 of the formula:
        # 1) Everybody gets a small base chance.
        # 2) First two hunters add stronger value.
        # 3) Hunters after the second still help, but less (diminishing returns).
        # 4) A hard cap prevents near-guaranteed hunts.
        hunter_count = len(hunters)
        primary_hunters = min(hunter_count, HUNT_SUCCESS_DIMINISH_AFTER)
        extra_hunters = max(0, hunter_count - HUNT_SUCCESS_DIMINISH_AFTER)
        skill = (
            HUNT_SUCCESS_BASE
            + primary_hunters * HUNT_SUCCESS_PER_HUNTER
            + extra_hunters * HUNT_SUCCESS_EXTRA_HUNTER
        )
        success_chance = min(HUNT_SUCCESS_CAP, skill)

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
            # Hunters earn trust from the party
            for agent in living:
                for hunter in hunters:
                    if agent is not hunter:
                        _adjust_relationship(agent, hunter.name, +0.05)

            # Starvation-behavior feedback:
            # ELI5: if the group spends a whole day hunting but brings back less
            # food (lbs) than there are people alive, it feels like a poor trade.
            # We apply a small morale penalty so agents are less likely to get
            # trapped repeating low-yield hunts forever.
            if food_gained < float(n):
                msgs.append(
                    "The catch is too small for the whole party; spirits dip slightly."
                )
                for agent in living:
                    agent.morale = max(0.0, agent.morale - 1.0)
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
            # Slight loss of trust in hunters
            for agent in living:
                for hunter in hunters:
                    if agent is not hunter:
                        _adjust_relationship(agent, hunter.name, -0.03)

            # The same low-yield lesson also applies to consolation foraging.
            if consolation_food < float(n):
                msgs.append(
                    "Foraging barely feeds the group; morale slips a little more."
                )
                for agent in living:
                    agent.morale = max(0.0, agent.morale - 1.0)
        return msgs

    def _do_repair(
        self, living: List["Agent"], world: "WagonTrain", n: int
    ) -> List[str]:
        msgs: List[str] = []
        from .agent import Role
        wheelwrights = [a for a in living if a.role == Role.WHEELWRIGHT]
        blacksmiths = [a for a in living if a.role == Role.BLACKSMITH]
        bonus = len(wheelwrights) * 5.0 + len(blacksmiths) * 3.0
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
        # Risk varies; scouts reduce danger.
        #
        # ELI5:
        # - Start with a baseline crossing danger (FORD_RISK_BASE).
        # - Each scout trims that danger a little bit.
        # - Keep a floor so crossings are never perfectly safe.
        from .agent import Role
        scouts = [a for a in living if a.role == Role.SCOUT]
        risk = max(
            FORD_RISK_FLOOR,
            FORD_RISK_BASE - len(scouts) * FORD_RISK_PER_SCOUT_REDUCTION,
        )

        # ELI5: later trail years had better crossing infrastructure and shared
        # route knowledge, so we scale risk down a bit by era.
        risk *= world.infrastructure_safety_factor
        if world.crossing_safety_days > 0:
            # ELI5: temporary bonus from finding safer crossing info or local
            # ferry/bridge tips in recent events.
            risk *= 0.8
            msgs.append(
                "Recent crossing advice and safer approaches reduce river danger today."
            )
        risk = max(FORD_RISK_FLOOR, risk)

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
            # Safe crossing can also improve confidence and preserve equipment.
            world.morale = min(100.0, world.morale + 3.0)
            world.wagon_parts = min(100.0, world.wagon_parts + 2.0)

        world.miles_traveled += FORD_MILES
        world.river_crossed()
        return msgs


    def _buy_supplies_at_fort_if_possible(self, world: "WagonTrain", msgs: List[str]) -> None:
        """Automatically buy essential supplies at fort stops using party cash."""
        if not world.at_fort_stop or world.cash <= 0.0:
            return

        max_food_to_buy = max(0.0, 700.0 - world.food_supply)
        affordable_food = world.cash / FORT_FOOD_PRICE_PER_LB
        food_to_buy = min(max_food_to_buy, affordable_food)
        if food_to_buy > 0.0:
            cost = food_to_buy * FORT_FOOD_PRICE_PER_LB
            world.food_supply += food_to_buy
            world.cash = max(0.0, world.cash - cost)
            msgs.append(f"Fort trade: bought {food_to_buy:.1f} lbs food for ${cost:.1f}.")

        max_parts_to_buy = max(0.0, 100.0 - world.wagon_parts)
        affordable_parts = world.cash / FORT_PARTS_PRICE_PER_POINT
        parts_to_buy = min(max_parts_to_buy, affordable_parts)
        if parts_to_buy > 0.0:
            cost = parts_to_buy * FORT_PARTS_PRICE_PER_POINT
            world.wagon_parts += parts_to_buy
            world.cash = max(0.0, world.cash - cost)
            msgs.append(f"Fort trade: bought {parts_to_buy:.1f} wagon parts for ${cost:.1f}.")

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

            # Emergency starvation behavior: slaughter one draft animal for meat.
            # ELI5: this saves lives now, but hurts travel speed later.
            slaughter_result = world.slaughter_draft_animal_for_food()
            if slaughter_result is not None:
                animal_type, meat_lbs = slaughter_result
                world.food_supply += meat_lbs
                msgs.append(
                    f"Emergency slaughter: one {animal_type} butchered for {meat_lbs:.1f} lbs of meat. "
                    "Mobility will suffer from the lost draft power."
                )
                # Important balancing nuance:
                # ELI5: they *were* starving this morning, but by evening they
                # now have fresh meat. So we apply a smaller same-day penalty
                # than a full no-food day.
                for agent in living:
                    agent.hunger = min(100.0, agent.hunger + 5.0)
                    agent.health -= 2.0
            else:
                # True no-food day with no slaughter fallback remains severe.
                for agent in living:
                    agent.hunger = min(100.0, agent.hunger + 15.0)
                    agent.health -= 10.0
        else:
            for agent in living:
                agent.hunger = max(0.0, agent.hunger - FOOD_PER_PERSON_PER_DAY * factor * 5)
        return msgs

    def _apply_sunday_work_penalty(self, living: List["Agent"]) -> List[str]:
        """Apply reduced-rest costs when the group works on Sunday.

        ELI5:
        - Working on Sunday is allowed.
        - But workers lose the equivalent of about half a rest day.
        - That means a small immediate health/morale decline.
        """
        msgs = [
            "Sunday work continues, but everyone only gets half rest and feels more worn down.",
        ]
        for agent in living:
            agent.health -= random.uniform(
                SUNDAY_WORK_HEALTH_LOSS_MIN,
                SUNDAY_WORK_HEALTH_LOSS_MAX,
            )
            agent.morale -= random.uniform(
                SUNDAY_WORK_MORALE_LOSS_MIN,
                SUNDAY_WORK_MORALE_LOSS_MAX,
            )
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

        # Captains improve cohesion and help morale recover from pressure.
        from .agent import Role
        captains = [a for a in living if a.role == Role.CAPTAIN]
        if captains:
            world.morale = min(100.0, world.morale + len(captains) * 1.2)
            for agent in living:
                for captain in captains:
                    if agent is not captain:
                        _adjust_relationship(agent, captain.name, +0.01)

        # Sync individual morale toward group morale slightly
        for agent in living:
            agent.morale += (world.morale - agent.morale) * 0.05
