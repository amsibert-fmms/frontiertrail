"""Unit tests for the wagon train multi-agent simulator."""

import sys
import os
import random
from datetime import date

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wagon_train.agent import Agent, Role, Traits
from wagon_train.world import LANDMARKS, TRAIL_DESTINATION, TRAIL_STOPS, WagonTrain, Weather
from wagon_train.decisions import Action, DecisionEngine, HUNT_FOOD_FAIL_MIN
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

    def test_effective_influence_for_world_boosts_needed_roles(self):
        # ELI5: specialists should get a moderate influence bump when their
        # specialty is needed in the current world state.
        world = WagonTrain()
        medic = Agent("Medic", Role.MEDIC, Traits())
        baseline = medic.effective_influence_for_world(world)

        world.sickness_events.append("Fever")
        boosted = medic.effective_influence_for_world(world)

        assert boosted > baseline

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

    def test_stagnation_panic_adds_travel_bias_after_three_low_progress_days(self):
        # ELI5: once low progress happens 3 days in a row, travel should get
        # an explicit scoring bonus so the group is more likely to break loops.
        world = WagonTrain()
        a = Agent("Traveler", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5), morale=60.0)

        world.low_progress_streak = 2
        score_before = a._action_score(world, Action.TRAVEL)
        world.low_progress_streak = 3
        score_after = a._action_score(world, Action.TRAVEL)

        assert score_after >= score_before + 2.0

    def test_fort_proximity_bias_increases_travel_and_reduces_hunt_and_repair(self):
        # ELI5: when a fort is the next stop and very close, travel should gain
        # urgency while hunt/repair become slightly less appealing.
        world = WagonTrain()
        a = Agent("Traveler", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5), morale=60.0)

        # Fort Kearny is at mile 320, so 300 miles traveled means 20 miles away.
        world.miles_traveled = 300.0
        travel_score = a._action_score(world, Action.TRAVEL)
        hunt_score = a._action_score(world, Action.HUNT)
        repair_score = a._action_score(world, Action.REPAIR_WAGON)

        # Move back far enough that the next stop is still Fort Kearny but outside
        # all urgency thresholds.
        world.miles_traveled = 180.0
        far_travel_score = a._action_score(world, Action.TRAVEL)
        far_hunt_score = a._action_score(world, Action.HUNT)
        far_repair_score = a._action_score(world, Action.REPAIR_WAGON)

        assert travel_score > far_travel_score
        assert hunt_score < far_hunt_score
        assert repair_score < far_repair_score

    def test_urgent_repair_assignee_forces_repair_vote(self):
        world = WagonTrain()
        a = Agent("Fixer", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5))
        world.urgent_repair_assignee = "Fixer"
        assert a.propose_action(world) == Action.REPAIR_WAGON

    def test_preacher_prefers_rest_on_sunday_when_stable(self):
        world = WagonTrain(start_year=1843)
        world.advance_day()
        while not world.is_sunday:
            world.advance_day()
        preacher = Agent("Rev", Role.PREACHER, Traits(0.5, 0.5, 0.5, 0.5), health=95.0, morale=80.0)
        assert preacher.propose_action(world) == Action.REST


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

    def test_calendar_starts_mar_1_and_year_is_in_requested_range(self):
        world = WagonTrain()
        world.advance_day()
        assert world.current_date.month == 3
        assert world.current_date.day == 1
        assert 1840 <= world.current_date.year <= 1860

    def test_start_year_can_be_forced_for_deterministic_date(self):
        world = WagonTrain(start_year=1851)
        world.advance_day()
        assert world.current_date == date(1851, 3, 1)

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
        world.miles_traveled = world.GOAL_MILES
        assert world.is_finished



    def test_goal_miles_matches_ordered_trail_data(self):
        world = WagonTrain()
        # ELI5: finish line should exactly match the final cumulative-mile stop.
        assert world.GOAL_MILES == 2040
        assert world.GOAL_MILES == TRAIL_STOPS[-1]["distance"]

    def test_next_landmark_tracks_progress(self):
        world = WagonTrain()
        # At the very start, the first stop ahead should be Kansas River Crossing.
        assert world.next_landmark == "Kansas River Crossing"

    def test_distance_to_next_landmark_tracks_remaining_miles(self):
        world = WagonTrain()
        # Kansas River Crossing is 102 miles from the start.
        assert world.distance_to_next_landmark == 102.0

        world.miles_traveled = 100.0
        assert world.distance_to_next_landmark == 2.0

        world.miles_traveled = world.GOAL_MILES
        assert world.distance_to_next_landmark == 0.0

        # At exactly 102 miles, we should now be aiming for Big Blue River Crossing.
        world.miles_traveled = 102.0
        assert world.next_landmark == "Big Blue River Crossing"

        # Near the end of the route, the next stop should be Barlow Road.
        world.miles_traveled = 1901.0
        assert world.next_landmark == "Barlow Road"

        # Once at/after destination, keep returning destination node.
        world.miles_traveled = world.GOAL_MILES
        assert world.next_landmark == TRAIL_DESTINATION

    def test_landmarks_contains_linear_hops_with_segment_miles(self):
        # ELI5: each hop should match the cumulative distance differences.
        assert ("Kansas River Crossing", 102) in LANDMARKS["Independence, Missouri"]["next"]
        assert ("Fort Kearny", 135) in LANDMARKS["Big Blue River Crossing"]["next"]
        assert ("The Dalles", 50) in LANDMARKS["Columbia River"]["next"]
        assert ("Oregon City / Willamette Valley", 60) in LANDMARKS["Barlow Road"]["next"]

    def test_stop_type_helpers_by_name(self):
        world = WagonTrain()
        # ELI5: these checks prove the stop labels were wired correctly.
        assert world.is_fort_stop("Fort Kearny")
        assert not world.is_fort_stop("Chimney Rock")

        assert world.is_river_crossing_stop("Green River Crossing")
        assert not world.is_river_crossing_stop("Fort Hall")

        assert world.is_landmark_stop("Blue Mountains")
        assert not world.is_landmark_stop("Soda Springs")

        # Unknown names should fail safely (False, not crash).
        assert not world.is_fort_stop("Not A Real Stop")

    def test_current_location_type_helpers(self):
        world = WagonTrain()

        # Start of trail is Independence, not tagged fort/river/landmark.
        assert world.current_or_last_stop == "Independence, Missouri"
        assert not world.at_fort_stop
        assert not world.at_river_crossing_stop
        assert not world.at_landmark_stop

        # At Fort Kearny miles, we should report fort context.
        world.miles_traveled = 320
        assert world.current_or_last_stop == "Fort Kearny"
        assert world.at_fort_stop

        # At Green River miles, we should report river crossing context.
        world.miles_traveled = 989
        assert world.current_or_last_stop == "Green River Crossing"
        assert world.at_river_crossing_stop

        # At Blue Mountains miles, we should report landmark context.
        world.miles_traveled = 1700
        assert world.current_or_last_stop == "Blue Mountains"
        assert world.at_landmark_stop

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

    def test_travel_passive_hunter_food_and_mechanic_repairs_apply(self, monkeypatch):
        # ELI5: on travel days hunters should gather a little food and mechanics
        # should recover a little wagon condition without spending a full action.
        engine = DecisionEngine()
        world = WagonTrain()
        world.river_ahead = False
        world.weather = Weather.SUNNY
        world.food_supply = 100.0
        world.wagon_parts = 50.0
        party = [
            Agent("Hunter1", Role.HUNTER, Traits()),
            Agent("Hunter2", Role.HUNTER, Traits()),
            Agent("Mech", Role.MECHANIC, Traits()),
        ]

        # Keep travel-distance randomness deterministic.
        monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2.0)
        outcomes = engine.apply_action(Action.TRAVEL, party, world)

        assert any("trail food" in msg for msg in outcomes)
        assert any("rolling maintenance" in msg for msg in outcomes)
        # Net food rises by +3.0 trail food then is reduced by travel consumption 4.5.
        assert world.food_supply == pytest.approx(98.5, abs=0.01)
        # +2 maintenance from one mechanic.
        assert world.wagon_parts >= 52.0

    def test_rest_with_medic_increases_healing_rate(self, monkeypatch):
        # ELI5: medics should make each rest day heal more than non-medic rests.
        engine = DecisionEngine()
        world = WagonTrain()
        with_medic = [
            Agent("Medic", Role.MEDIC, Traits(), health=50.0),
            Agent("Traveler", Role.PASSENGER, Traits(), health=50.0),
        ]
        without_medic = [
            Agent("Traveler1", Role.PASSENGER, Traits(), health=50.0),
            Agent("Traveler2", Role.PASSENGER, Traits(), health=50.0),
        ]

        monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2.0)
        engine.apply_action(Action.REST, with_medic, world)
        healed_with_medic = sum(a.health for a in with_medic) / len(with_medic)

        world2 = WagonTrain()
        engine.apply_action(Action.REST, without_medic, world2)
        healed_without_medic = sum(a.health for a in without_medic) / len(without_medic)

        assert healed_with_medic > healed_without_medic

    def test_preacher_adds_morale_on_successful_sunday_rest(self, monkeypatch):
        # ELI5: when Sunday rest is chosen and a preacher is present, morale gets
        # a visible extra bump.
        engine = DecisionEngine()
        world = WagonTrain(start_year=1848)
        world.advance_day()
        while not world.is_sunday:
            world.advance_day()
        world.morale = 60.0
        party = [
            Agent("Preacher", Role.PREACHER, Traits(), morale=50.0),
            Agent("Traveler", Role.PASSENGER, Traits(), morale=50.0),
        ]

        monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2.0)
        outcomes = engine.apply_action(Action.REST, party, world)

        assert any("Sunday rest with preacher support" in msg for msg in outcomes)
        assert world.morale > 63.0

    def test_collect_votes_uses_contextual_influence_boosts(self):
        # ELI5: medic should cast a bigger vote when sickness is active.
        engine = DecisionEngine()
        world = WagonTrain()
        world.sickness_events.append("Illness")

        medic = Agent("Medic", Role.MEDIC, Traits())
        passenger = Agent("Passenger", Role.PASSENGER, Traits())

        # Force both to vote REST so the tally is a simple sum.
        medic.propose_action = lambda w: Action.REST  # type: ignore[method-assign]
        passenger.propose_action = lambda w: Action.REST  # type: ignore[method-assign]

        tally = engine.collect_votes([medic, passenger], world)
        assert tally[Action.REST] > medic.effective_influence + passenger.effective_influence

    def test_work_on_sunday_applies_half_rest_penalty(self, monkeypatch):
        engine = DecisionEngine()
        world = WagonTrain(start_year=1848)
        world.advance_day()
        while not world.is_sunday:
            world.advance_day()

        party = self._make_party()
        for a in party:
            a.health = 90.0
            a.morale = 80.0

        # Keep numbers deterministic for stable assertions.
        monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2.0)
        outcomes = engine.apply_action(Action.TRAVEL, party, world)

        assert any("half rest" in msg for msg in outcomes)
        assert sum(a.health for a in party) / len(party) < 90.0
        assert sum(a.morale for a in party) / len(party) < 80.0

    def test_hunt_may_add_food(self):
        random.seed(42)
        engine = DecisionEngine()
        world = WagonTrain()
        party = [Agent("Hunter", Role.HUNTER, Traits())]
        food_before = world.food_supply
        engine.apply_action(Action.HUNT, party, world)
        # Food may increase or not — just ensure it doesn't go negative
        assert world.food_supply >= 0.0


    def test_hunt_low_yield_applies_extra_morale_penalty(self, monkeypatch):
        # ELI5: if hunt yield is smaller than party size, morale should dip a
        # little extra so the AI learns this is not a good repeated strategy.
        engine = DecisionEngine()
        world = WagonTrain()
        party = [Agent(f"A{i}", Role.PASSENGER, Traits()) for i in range(5)]
        for a in party:
            a.morale = 50.0

        # Force failed hunt with tiny consolation food below party size.
        monkeypatch.setattr(random, "random", lambda: 0.999)
        monkeypatch.setattr(random, "uniform", lambda a, b: 4.6 if a == HUNT_FOOD_FAIL_MIN else (a + b) / 2.0)
        outcomes = engine.apply_action(Action.HUNT, party, world)

        assert any("barely feeds" in msg for msg in outcomes)
        assert sum(a.morale for a in party) / len(party) < 48.5

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
        assert len(party) == 20
        roles = {a.role for a in party}
        assert Role.LEADER in roles
        assert Role.HUNTER in roles
        assert Role.MEDIC in roles
        assert Role.MECHANIC in roles
        assert Role.SCOUT in roles
        assert Role.PASSENGER in roles
        assert Role.PREACHER in roles

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
