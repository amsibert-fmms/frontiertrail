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
        world.food_supply = 4.0   # 4 / (1 * 1.5) ≈ 2.7 days < 5 threshold
        world.living_count = 1
        a = Agent("Hunter", Role.HUNTER, Traits(0.5, 0.5, 0.5, 0.5))
        # A hunter should suggest hunting when only ~2-3 days of food remain
        action = a.propose_action(world)
        assert action == Action.HUNT

    def test_hunter_proposes_hunt_when_food_sufficient(self):
        world = WagonTrain()
        world.food_supply = 500.0  # many days of food remaining
        world.living_count = 10
        a = Agent("Hunter", Role.HUNTER, Traits(0.5, 0.5, 0.5, 0.5))
        # With plenty of food (>5 days), hunter should NOT immediately hunt
        action = a.propose_action(world)
        assert action != Action.HUNT

    def test_mechanic_proposes_repair_when_parts_low(self):
        world = WagonTrain()
        world.wagon_parts = 10.0
        a = Agent("Mech", Role.MECHANIC, Traits(0.5, 0.5, 0.5, 0.5))
        action = a.propose_action(world)
        assert action == Action.REPAIR_WAGON

    def test_relationships_initialized_empty(self):
        a = Agent("Test", Role.PASSENGER, Traits())
        assert isinstance(a.relationships, dict)
        assert len(a.relationships) == 0  # empty until initialised by Simulation

    def test_relationship_modifier_neutral(self):
        a = Agent("Alice", Role.LEADER, Traits())
        b = Agent("Bob", Role.HUNTER, Traits())
        c = Agent("Carol", Role.MEDIC, Traits())
        # Bob and Carol both have neutral (0.5) trust toward Alice
        b.relationships["Alice"] = 0.5
        c.relationships["Alice"] = 0.5
        modifier = a.get_relationship_modifier([a, b, c])
        assert abs(modifier - 1.0) < 1e-6  # neutral trust → 1.0 modifier

    def test_relationship_modifier_high_trust(self):
        a = Agent("Alice", Role.LEADER, Traits())
        b = Agent("Bob", Role.HUNTER, Traits())
        b.relationships["Alice"] = 1.0  # full trust
        modifier = a.get_relationship_modifier([a, b])
        assert modifier == 1.5  # max modifier

    def test_relationship_modifier_low_trust(self):
        a = Agent("Alice", Role.LEADER, Traits())
        b = Agent("Bob", Role.HUNTER, Traits())
        b.relationships["Alice"] = 0.0  # full distrust
        modifier = a.get_relationship_modifier([a, b])
        assert modifier == 0.5  # min modifier

    def test_update_relationship_clamps(self):
        a = Agent("Alice", Role.LEADER, Traits())
        a.update_relationship("Bob", 0.3)
        assert a.relationships["Bob"] == pytest.approx(0.8)
        a.update_relationship("Bob", 0.5)
        assert a.relationships["Bob"] == 1.0  # clamped at max
        a.update_relationship("Bob", -2.0)
        assert a.relationships["Bob"] == 0.0  # clamped at min


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

    def test_food_days_remaining(self):
        world = WagonTrain()
        world.food_supply = 15.0
        world.living_count = 10  # 10 * 1.5 = 15 food/day → 1 day remaining
        assert abs(world.food_days_remaining - 1.0) < 1e-6

    def test_food_days_remaining_zero_people(self):
        world = WagonTrain()
        world.food_supply = 100.0
        world.living_count = 0  # no living agents → infinite days
        assert world.food_days_remaining == float("inf")

    def test_wagon_condition(self):
        world = WagonTrain()
        world.wagon_parts = 50.0
        assert world.wagon_condition == pytest.approx(0.5)
        world.wagon_parts = 100.0
        assert world.wagon_condition == pytest.approx(1.0)

    def test_recent_travel_speed_empty(self):
        world = WagonTrain()
        assert world.recent_travel_speed == 0.0


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

    def test_travel_tracks_recent_speed(self):
        engine = DecisionEngine()
        world = WagonTrain()
        party = self._make_party()
        assert world.recent_travel_speed == 0.0
        engine.apply_action(Action.TRAVEL, party, world)
        assert world.recent_travel_speed > 0.0

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

    def test_ford_river_crosses_when_river_present(self):
        engine = DecisionEngine()
        world = WagonTrain()
        world.river_ahead = True
        party = self._make_party()
        initial_miles = world.miles_traveled
        engine.apply_action(Action.FORD_RIVER, party, world)
        assert not world.river_ahead
        assert world.miles_traveled >= initial_miles

    def test_wait_at_river_keeps_river_ahead(self):
        engine = DecisionEngine()
        world = WagonTrain()
        world.river_ahead = True
        party = self._make_party()
        engine.apply_action(Action.WAIT_AT_RIVER, party, world)
        assert world.river_ahead  # river should still be there after waiting

    def test_caulk_wagon_crosses_river(self):
        random.seed(0)  # seed that avoids caulk failure
        engine = DecisionEngine()
        world = WagonTrain()
        world.river_ahead = True
        world.wagon_parts = 100.0
        party = self._make_party()
        engine.apply_action(Action.CAULK_WAGON, party, world)
        assert not world.river_ahead  # river crossed regardless of success/failure

    def test_ferry_across_crosses_river(self):
        engine = DecisionEngine()
        world = WagonTrain()
        world.river_ahead = True
        world.food_supply = 200.0  # enough to pay for ferry
        party = self._make_party()
        food_before = world.food_supply
        engine.apply_action(Action.FERRY_ACROSS, party, world)
        assert not world.river_ahead
        # Ferry should cost food (after daily consumption)
        assert world.food_supply < food_before

    def test_ferry_falls_back_to_ford_when_food_low(self):
        random.seed(1)
        engine = DecisionEngine()
        world = WagonTrain()
        world.river_ahead = True
        world.food_supply = 1.0  # too low to pay for ferry
        party = self._make_party()
        engine.apply_action(Action.FERRY_ACROSS, party, world)
        assert not world.river_ahead  # river still crossed via ford fallback

    def test_relationships_affect_vote_weight(self):
        """Agents with high trust earn more influence in voting."""
        engine = DecisionEngine()
        world = WagonTrain()
        world.river_ahead = False
        leader = Agent("Leader", Role.LEADER, Traits(0.5, 0.7, 0.6, 0.8))
        follower = Agent("Follower", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5))
        # Follower highly trusts leader
        follower.relationships["Leader"] = 1.0
        party = [leader, follower]
        tally_high = engine.collect_votes(party, world)

        leader2 = Agent("Leader", Role.LEADER, Traits(0.5, 0.7, 0.6, 0.8))
        follower2 = Agent("Follower", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5))
        # Follower distrusts leader2
        follower2.relationships["Leader"] = 0.0
        party2 = [leader2, follower2]
        tally_low = engine.collect_votes(party2, world)

        # Leader should have higher weighted vote when trusted
        leader_action_high = leader.propose_action(world)
        leader_action_low = leader2.propose_action(world)
        if leader_action_high == leader_action_low:
            assert tally_high.get(leader_action_high, 0) > tally_low.get(leader_action_low, 0)


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

    def test_simulation_initializes_relationships(self):
        sim = Simulation(seed=1, log_to_stdout=False)
        for agent in sim.agents:
            for other in sim.agents:
                if other is not agent:
                    assert other.name in agent.relationships
                    assert agent.relationships[other.name] == 0.5

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
