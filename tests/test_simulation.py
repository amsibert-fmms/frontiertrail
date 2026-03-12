"""Unit tests for the wagon train multi-agent simulator."""

import sys
import os
import random

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wagon_train.agent import Agent, Role, Traits
from wagon_train.world import WagonTrain, Weather
from wagon_train.decisions import Action, DecisionEngine
from wagon_train.events import EventSystem
from wagon_train.logger import SimulationLogger
from wagon_train.simulation import Simulation, build_default_party


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestTraits:
    def test_default_traits(self):
        t = Traits()
        assert t.risk_tolerance == 0.5
        assert t.generosity == 0.5

    def test_invalid_trait_raises(self):
        with pytest.raises(ValueError):
            Traits(risk_tolerance=1.5)

    def test_random_traits_in_range(self):
        for _ in range(20):
            t = Traits.random()
            for attr in ("risk_tolerance", "generosity", "stubbornness", "cooperation"):
                assert 0.0 <= getattr(t, attr) <= 1.0


class TestAgent:
    def _make_agent(self, role=Role.PASSENGER):
        return Agent("Test", role, Traits(0.5, 0.5, 0.5, 0.5))

    def test_health_clamped(self):
        a = self._make_agent()
        a.health = 200
        assert a.health == 100.0
        a.health = -50
        assert a.health == 0.0

    def test_death_on_zero_health(self):
        a = self._make_agent()
        a.health = 0
        assert not a.alive

    def test_hunger_clamped(self):
        a = self._make_agent()
        a.hunger = 150
        assert a.hunger == 100.0
        a.hunger = -10
        assert a.hunger == 0.0

    def test_morale_clamped(self):
        a = self._make_agent()
        a.morale = -5
        assert a.morale == 0.0
        a.morale = 105
        assert a.morale == 100.0

    def test_effective_influence_scales_with_health(self):
        a = Agent("Test", Role.LEADER, Traits())
        full_influence = a.effective_influence
        a.health = 50.0
        reduced_influence = a.effective_influence
        assert reduced_influence < full_influence

    def test_dead_agent_has_zero_influence(self):
        a = self._make_agent()
        a.alive = False
        assert a.effective_influence == 0.0

    def test_repr(self):
        a = self._make_agent()
        rep = repr(a)
        assert "Test" in rep
        assert "passenger" in rep

    def test_propose_action_returns_action(self):
        world = WagonTrain()
        a = Agent("Test", Role.LEADER, Traits(0.5, 0.5, 0.5, 0.5))
        action = a.propose_action(world)
        assert isinstance(action, Action)

    def test_hunter_proposes_hunt_when_food_low(self):
        world = WagonTrain()
        world.food_supply = 10.0
        a = Agent("Hunter", Role.HUNTER, Traits(0.5, 0.5, 0.5, 0.5))
        # A hunter should suggest hunting with low food supply
        action = a.propose_action(world)
        assert action == Action.HUNT

    def test_mechanic_proposes_repair_when_parts_low(self):
        world = WagonTrain()
        world.wagon_parts = 10.0
        a = Agent("Mech", Role.MECHANIC, Traits(0.5, 0.5, 0.5, 0.5))
        action = a.propose_action(world)
        assert action == Action.REPAIR_WAGON

    def test_progress_bias_prefers_travel_when_behind_schedule(self):
        # When the group is very behind schedule and not in emergency survival
        # territory, normal passengers should lean toward progress actions.
        world = WagonTrain()
        world.day = 150
        world.miles_traveled = 500.0
        a = Agent("Traveler", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5), morale=60.0)
        action = a.propose_action(world)
        assert action == Action.TRAVEL

    def test_urgent_repair_assignee_forces_repair_vote(self):
        world = WagonTrain()
        a = Agent("Fixer", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5))
        world.urgent_repair_assignee = "Fixer"
        assert a.propose_action(world) == Action.REPAIR_WAGON


# ---------------------------------------------------------------------------
# World / WagonTrain tests
# ---------------------------------------------------------------------------

class TestWagonTrain:
    def test_initial_state(self):
        world = WagonTrain()
        assert world.day == 0
        assert world.miles_traveled == 0.0
        assert world.food_supply == 500.0
        assert world.wagon_parts == 100.0
        assert world.morale == 80.0

    def test_advance_day_increments_day(self):
        world = WagonTrain()
        world.advance_day()
        assert world.day == 1

    def test_advance_day_reduces_food(self):
        world = WagonTrain()
        food_before = world.food_supply
        world.advance_day()
        assert world.food_supply < food_before

    def test_advance_day_reduces_parts(self):
        world = WagonTrain()
        parts_before = world.wagon_parts
        world.advance_day()
        assert world.wagon_parts <= parts_before

    def test_is_finished_on_miles(self):
        world = WagonTrain()
        world.miles_traveled = 2000.0
        assert world.is_finished

    def test_is_finished_on_days(self):
        world = WagonTrain()
        world.day = 200
        assert world.is_finished

    def test_not_finished_initially(self):
        world = WagonTrain()
        assert not world.is_finished

    def test_sickness_count(self):
        world = WagonTrain()
        assert world.sickness_count == 0
        world.sickness_events.append("Test illness")
        assert world.sickness_count == 1

    def test_river_crossed_resets_state(self):
        world = WagonTrain()
        world.river_ahead = True
        world.river_crossed()
        assert not world.river_ahead

    def test_travel_modifier_for_weather(self):
        world = WagonTrain(weather=Weather.STORMY)
        assert world.travel_modifier < 1.0

    def test_miles_needed_per_remaining_day_is_positive(self):
        world = WagonTrain()
        world.day = 100
        world.miles_traveled = 900
        assert world.miles_needed_per_remaining_day > 0

    def test_record_daily_progress_clamps_negative_and_tracks_streak(self):
        world = WagonTrain()
        world.record_daily_progress(-10.0)
        assert world.last_daily_progress == 0.0
        assert world.low_progress_streak == 1
        world.record_daily_progress(12.0)
        assert world.low_progress_streak == 0


# ---------------------------------------------------------------------------
# Decision engine tests
# ---------------------------------------------------------------------------

class TestDecisionEngine:
    def _make_party(self):
        return [
            Agent(f"Agent{i}", Role.PASSENGER, Traits()) for i in range(5)
        ]

    def test_resolve_returns_action(self):
        engine = DecisionEngine()
        party = self._make_party()
        world = WagonTrain()
        action, tally = engine.resolve(party, world)
        assert isinstance(action, Action)
        assert isinstance(tally, dict)

    def test_resolve_no_living_defaults_to_rest(self):
        engine = DecisionEngine()
        party = self._make_party()
        for a in party:
            a.alive = False
        world = WagonTrain()
        action, tally = engine.resolve(party, world)
        assert action == Action.REST

    def test_travel_increases_miles(self):
        engine = DecisionEngine()
        world = WagonTrain()
        party = self._make_party()
        initial_miles = world.miles_traveled
        engine.apply_action(Action.TRAVEL, party, world)
        assert world.miles_traveled > initial_miles

    def test_rest_increases_health(self):
        engine = DecisionEngine()
        world = WagonTrain()
        party = self._make_party()
        for a in party:
            a.health = 50.0
        engine.apply_action(Action.REST, party, world)
        avg_health = sum(a.health for a in party) / len(party)
        assert avg_health > 50.0

    def test_hunt_may_add_food(self):
        random.seed(42)
        engine = DecisionEngine()
        world = WagonTrain()
        party = [Agent("Hunter", Role.HUNTER, Traits())]
        food_before = world.food_supply
        engine.apply_action(Action.HUNT, party, world)
        # Food may increase or not — just ensure it doesn't go negative
        assert world.food_supply >= 0.0

    def test_hunt_failure_still_adds_small_food(self, monkeypatch):
        # Force hunt failure and verify consolation foraging food is added.
        engine = DecisionEngine()
        world = WagonTrain()
        world.weather = Weather.SUNNY
        party = [Agent("Hunter", Role.HUNTER, Traits())]
        before_food = world.food_supply

        # First random.random call in _do_hunt decides success/failure.
        monkeypatch.setattr(random, "random", lambda: 0.999)
        engine.apply_action(Action.HUNT, party, world)

        # Even after daily food consumption, failed hunts should not produce
        # a net catastrophic food drop from zero gain; foraging adds some food.
        assert world.food_supply > before_food - 2.0

    def test_repair_increases_parts(self):
        engine = DecisionEngine()
        world = WagonTrain()
        world.wagon_parts = 20.0
        party = [Agent("Mech", Role.MECHANIC, Traits())]
        engine.apply_action(Action.REPAIR_WAGON, party, world)
        assert world.wagon_parts > 20.0

    def test_ration_reduces_food_consumption(self):
        """Ration food action should consume less food than normal travel."""
        engine = DecisionEngine()
        world1 = WagonTrain()
        world2 = WagonTrain()
        party1 = self._make_party()
        party2 = self._make_party()

        engine.apply_action(Action.RATION_FOOD, party1, world1)
        engine.apply_action(Action.TRAVEL, party2, world2)

        # After ration, food supply should be higher (less consumed)
        assert world1.food_supply >= world2.food_supply

    def test_repair_clears_urgent_repair_assignment(self):
        engine = DecisionEngine()
        world = WagonTrain()
        world.urgent_repair_assignee = "Mech"
        party = [Agent("Mech", Role.MECHANIC, Traits())]
        engine.apply_action(Action.REPAIR_WAGON, party, world)
        assert world.urgent_repair_assignee is None

    def test_ford_river_crosses_when_river_present(self):
        engine = DecisionEngine()
        world = WagonTrain()
        world.river_ahead = True
        party = self._make_party()
        initial_miles = world.miles_traveled
        engine.apply_action(Action.FORD_RIVER, party, world)
        assert not world.river_ahead
        assert world.miles_traveled >= initial_miles


# ---------------------------------------------------------------------------
# Event system tests
# ---------------------------------------------------------------------------

class TestEventSystem:
    def test_roll_returns_list(self):
        es = EventSystem()
        world = WagonTrain()
        party = [Agent("A", Role.PASSENGER, Traits())]
        result = es.roll(world, party)
        assert isinstance(result, list)

    def test_event_can_affect_food(self):
        random.seed(7)  # seed that triggers a food-affecting event
        es = EventSystem()
        for _ in range(100):
            world = WagonTrain()
            party = [Agent("A", Role.PASSENGER, Traits())]
            before = world.food_supply
            es.roll(world, party)
            # Just verify no crashes and values stay non-negative
            assert world.food_supply >= 0.0

    def test_negative_miles_event_never_moves_backwards(self):
        es = EventSystem()
        world = WagonTrain()
        world.miles_traveled = 100.0
        party = [Agent("A", Role.MECHANIC, Traits())]

        synthetic = {
            "name": "Synthetic Stall",
            "description": "Testing no-backwards-travel rule.",
            "food_delta": 0,
            "parts_delta": 0,
            "health_delta": 0,
            "morale_delta": 0,
            "miles_delta": -25,
            "sickness": False,
        }
        es._choose_event = lambda: synthetic  # type: ignore[method-assign]
        es.BASE_EVENT_CHANCE = 1.0
        es.roll(world, party)
        assert world.miles_traveled == 100.0

    def test_wagon_break_event_assigns_urgent_repair_owner(self):
        es = EventSystem()
        world = WagonTrain()
        party = [
            Agent("Mechanic", Role.MECHANIC, Traits()),
            Agent("Passenger", Role.PASSENGER, Traits()),
        ]
        synthetic = {
            "name": "Synthetic Wagon Break",
            "description": "Testing urgent repair assignment.",
            "food_delta": 0,
            "parts_delta": -10,
            "health_delta": 0,
            "morale_delta": 0,
            "sickness": False,
            "wagon_break": True,
        }
        es._choose_event = lambda: synthetic  # type: ignore[method-assign]
        es.BASE_EVENT_CHANCE = 1.0
        es.roll(world, party)
        assert world.urgent_repair_assignee == "Mechanic"


# ---------------------------------------------------------------------------
# Logger tests
# ---------------------------------------------------------------------------

class TestSimulationLogger:
    def test_history_populated(self):
        logger = SimulationLogger(log_to_stdout=False)
        world = WagonTrain()
        world.advance_day()
        logger.log_day_header(world)
        assert len(logger.history) > 0

    def test_log_outcomes(self):
        logger = SimulationLogger(log_to_stdout=False)
        logger.log_outcomes(["Something happened"])
        assert any("Something happened" in h for h in logger.history)


# ---------------------------------------------------------------------------
# Simulation integration test
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_build_default_party(self):
        party = build_default_party()
        assert len(party) == 10
        roles = {a.role for a in party}
        assert Role.LEADER in roles
        assert Role.HUNTER in roles
        assert Role.MEDIC in roles
        assert Role.MECHANIC in roles
        assert Role.SCOUT in roles
        assert Role.PASSENGER in roles

    def test_simulation_terminates(self):
        """Simulation should always terminate within 200 days."""
        sim = Simulation(seed=123, log_to_stdout=False)
        reason = sim.run()
        assert isinstance(reason, str)
        assert sim.world.day <= 200

    def test_simulation_reaches_or_stops(self):
        sim = Simulation(seed=456, log_to_stdout=False)
        sim.run()
        assert sim.world.is_finished or not any(a.alive for a in sim.agents)

    def test_miles_non_negative(self):
        sim = Simulation(seed=789, log_to_stdout=False)
        sim.run()
        assert sim.world.miles_traveled >= 0.0

    def test_reproducible_with_seed(self):
        sim1 = Simulation(seed=999, log_to_stdout=False)
        reason1 = sim1.run()

        sim2 = Simulation(seed=999, log_to_stdout=False)
        reason2 = sim2.run()

        assert reason1 == reason2
        assert sim1.world.miles_traveled == sim2.world.miles_traveled
        assert sim1.world.day == sim2.world.day
