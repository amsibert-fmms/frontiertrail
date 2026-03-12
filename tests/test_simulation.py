"""Unit tests for the wagon train multi-agent simulator."""

import sys
import os
import random
from datetime import date

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wagon_train.agent import Agent, Role, Traits
from wagon_train.world import (
    LANDMARKS,
    TERRAIN_SPEED_SEGMENTS,
    TRAIL_DESTINATION,
    TRAIL_STOPS,
    WagonTrain,
    Weather,
)
from wagon_train.decisions import Action, DecisionEngine, HUNT_FOOD_FAIL_MIN
from wagon_train.events import EventSystem
from wagon_train.logger import SimulationLogger
from wagon_train.simulation import PASSENGER_FARE_DOLLARS, Simulation, build_default_party


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
        a = Agent("Test", Role.CAPTAIN, Traits())
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
        a = Agent("Test", Role.CAPTAIN, Traits(0.5, 0.5, 0.5, 0.5))
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

    def test_wheelwright_proposes_repair_when_parts_low(self):
        world = WagonTrain()
        world.wagon_parts = 10.0
        a = Agent("Wheel", Role.WHEELWRIGHT, Traits(0.5, 0.5, 0.5, 0.5))
        action = a.propose_action(world)
        assert action == Action.REPAIR_WAGON

    def test_relationships_initialized_empty(self):
        a = Agent("Test", Role.PASSENGER, Traits())
        assert isinstance(a.relationships, dict)
        assert len(a.relationships) == 0  # empty until initialised by Simulation

    def test_relationship_modifier_neutral(self):
        a = Agent("Alice", Role.CAPTAIN, Traits())
        b = Agent("Bob", Role.HUNTER, Traits())
        c = Agent("Carol", Role.MEDIC, Traits())
        # Bob and Carol both have neutral (0.5) trust toward Alice
        b.relationships["Alice"] = 0.5
        c.relationships["Alice"] = 0.5
        modifier = a.get_relationship_modifier([a, b, c])
        assert abs(modifier - 1.0) < 1e-6  # neutral trust → 1.0 modifier

    def test_relationship_modifier_high_trust(self):
        a = Agent("Alice", Role.CAPTAIN, Traits())
        b = Agent("Bob", Role.HUNTER, Traits())
        b.relationships["Alice"] = 1.0  # full trust
        modifier = a.get_relationship_modifier([a, b])
        assert modifier == 1.5  # max modifier

    def test_relationship_modifier_low_trust(self):
        a = Agent("Alice", Role.CAPTAIN, Traits())
        b = Agent("Bob", Role.HUNTER, Traits())
        b.relationships["Alice"] = 0.0  # full distrust
        modifier = a.get_relationship_modifier([a, b])
        assert modifier == 0.5  # min modifier

    def test_update_relationship_clamps(self):
        a = Agent("Alice", Role.CAPTAIN, Traits())
        a.update_relationship("Bob", 0.3)
        assert a.relationships["Bob"] == pytest.approx(0.8)
        a.update_relationship("Bob", 0.5)
        assert a.relationships["Bob"] == 1.0  # clamped at max
        a.update_relationship("Bob", -2.0)
        assert a.relationships["Bob"] == 0.0  # clamped at min
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

    def test_cook_role_can_be_instantiated(self):
        world = WagonTrain(start_year=1843)
        world.advance_day()
        while not world.is_sunday:
            world.advance_day()
        cook = Agent("Rev", Role.COOK, Traits(0.5, 0.5, 0.5, 0.5), health=95.0, morale=80.0)
        assert cook.role == Role.COOK

    def test_agents_start_with_route_destination_intent(self):
        # ELI5: each traveler starts with a destination in mind before voting.
        traveler = Agent("Traveler", Role.PASSENGER, Traits(0.6, 0.5, 0.5, 0.5))
        assert traveler.intended_route in {"oregon", "california"}

    def test_exceptional_hunger_can_shift_route_preference_to_california(self):
        # ELI5: this traveler starts Oregon-leaning, but severe hunger can flip
        # to a shorter-path preference in our model.
        world = WagonTrain()
        world.living_count = 1
        world.food_supply = 2.0
        medic = Agent("Medic", Role.MEDIC, Traits(0.2, 0.5, 0.1, 0.5), hunger=90.0)
        medic.intended_route = "oregon"

        assert medic.preferred_trail(world) == "california"

    def test_sickness_and_exhaustion_shift_preference_to_oregon(self):
        # ELI5: even California-leaning travelers should pick easier routing
        # when very sick and exhausted.
        world = WagonTrain()
        world.sickness_events = ["fever", "infection"]
        world.morale = 35.0
        scout = Agent("Scout", Role.SCOUT, Traits(0.9, 0.5, 0.1, 0.5), health=30.0, morale=20.0)
        scout.intended_route = "california"

        assert scout.preferred_trail(world) == "oregon"



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
        # ELI5: default team starts with a full set of pull animals.
        assert world.oxen_count == 8
        assert world.horse_count == 2
        assert world.mule_count == 2

    def test_summary_includes_animals_and_day_header_once(self):
        world = WagonTrain(start_year=1848)
        world.advance_day()
        summary = world.summary()
        # Regression check: "Day" header should only appear once.
        assert summary.count("Day") == 1
        assert "Animals: O" in summary

    def test_slaughter_draft_animal_prefers_oxen(self):
        world = WagonTrain(oxen_count=1, horse_count=1, mule_count=1)
        slaughter = world.slaughter_draft_animal_for_food()
        assert slaughter is not None
        animal_type, meat_lbs = slaughter
        assert animal_type == "ox"
        assert 400.0 <= meat_lbs <= 600.0
        assert world.oxen_count == 0

    def test_advance_day_increments_day(self):
        world = WagonTrain()
        world.advance_day()
        assert world.day == 1

    def test_calendar_starts_in_apr_or_may_and_year_is_in_requested_range(self):
        world = WagonTrain()
        world.advance_day()
        assert world.current_date.month in (4, 5)
        assert world.current_date.day == 1
        assert 1841 <= world.current_date.year <= 1860

    def test_start_year_can_be_forced_for_deterministic_date(self):
        world = WagonTrain(start_year=1851, start_month=4)
        world.advance_day()
        assert world.current_date == date(1851, 4, 1)

    def test_invalid_start_month_raises(self):
        with pytest.raises(ValueError):
            WagonTrain(start_month=3)

    def test_infrastructure_safety_factor_improves_in_later_years(self):
        # ELI5: later trail years should have better crossing safety due to
        # more ferries/bridges and shared route knowledge.
        early = WagonTrain(start_year=1842, start_month=4)
        late = WagonTrain(start_year=1855, start_month=4)
        assert late.infrastructure_safety_factor < early.infrastructure_safety_factor

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


    def test_trail_choice_switches_route_context(self):
        world = WagonTrain()
        # ELI5: reaching Soda Springs should unlock the branch vote.
        world.miles_traveled = 1180.0
        assert world.needs_trail_choice

        world.apply_trail_choice("california")

        assert world.trail_choice_made
        assert world.active_route == "california"
        assert world.next_landmark == "Bear River Divide"

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

    def test_terrain_speed_modifier_uses_segment_ranges(self):
        world = WagonTrain()

        # ELI5: right after starting, we are in plains terrain.
        world.miles_traveled = 10.0
        assert world.terrain_segment_name == "plains"
        assert world.terrain_speed_modifier == pytest.approx(1.0)

        # ELI5: around South Pass / Rocky region, travel should slow.
        world.miles_traveled = 950.0
        assert world.terrain_segment_name == "mountains"
        assert world.terrain_speed_modifier == pytest.approx(0.82)

        # ELI5: farther west through desert-like terrain, still slower.
        world.miles_traveled = 1450.0
        assert world.terrain_segment_name == "desert"
        assert world.terrain_speed_modifier == pytest.approx(0.78)

        # ELI5: late trail forests recover slightly versus desert.
        world.miles_traveled = 1750.0
        assert world.terrain_segment_name == "forests"
        assert world.terrain_speed_modifier == pytest.approx(0.9)

    def test_terrain_segments_do_not_overlap_and_cover_positive_miles(self):
        # ELI5: this protects against misconfigured segment tables.
        # We validate ranges are ordered and have positive width.
        previous_end = 0.0
        for _name, start_mile, end_mile, _multiplier in TERRAIN_SPEED_SEGMENTS:
            assert start_mile >= previous_end
            assert end_mile > start_mile
            previous_end = end_mile

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

    def test_travel_passive_hunter_food_and_wheelwright_repairs_apply(self, monkeypatch):
        # ELI5: on travel days hunters should gather a little food and wheelwrights
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
            Agent("Wheel", Role.WHEELWRIGHT, Traits()),
        ]

        # Keep travel-distance randomness deterministic.
        monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2.0)
        outcomes = engine.apply_action(Action.TRAVEL, party, world)

        assert any("trail food" in msg for msg in outcomes)
        assert any("rolling maintenance" in msg for msg in outcomes)
        # Net food rises by +3.0 hunter food and +0.96 general foraging, then
        # is reduced by travel consumption 4.5.
        assert world.food_supply == pytest.approx(99.46, abs=0.01)
        # +2 maintenance from one wheelwright.
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

    def test_cook_adds_extra_rest_recovery(self, monkeypatch):
        # ELI5: cooks should make rest actions a little better at recovery.
        engine = DecisionEngine()
        world = WagonTrain()
        with_cook = [
            Agent("Cook", Role.COOK, Traits(), health=50.0),
            Agent("Traveler", Role.PASSENGER, Traits(), health=50.0),
        ]
        without_cook = [
            Agent("Traveler1", Role.PASSENGER, Traits(), health=50.0),
            Agent("Traveler2", Role.PASSENGER, Traits(), health=50.0),
        ]

        monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2.0)
        engine.apply_action(Action.REST, with_cook, world)
        healed_with_cook = sum(a.health for a in with_cook) / len(with_cook)

        world2 = WagonTrain()
        engine.apply_action(Action.REST, without_cook, world2)
        healed_without_cook = sum(a.health for a in without_cook) / len(without_cook)

        assert healed_with_cook > healed_without_cook

    def test_travel_buys_supplies_at_fort_when_cash_available(self, monkeypatch):
        engine = DecisionEngine()
        world = WagonTrain()
        world.miles_traveled = 320.0  # Fort Kearny
        world.cash = 40.0
        world.food_supply = 620.0
        world.wagon_parts = 90.0
        party = [Agent("Traveler", Role.PASSENGER, Traits())]

        monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2.0)
        outcomes = engine.apply_action(Action.TRAVEL, party, world)

        assert any("Fort trade" in msg for msg in outcomes)
        assert world.cash < 40.0
        assert world.food_supply > 620.0

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


    def test_seeded_route_sweep_shifts_with_distress_profile(self):
        # ELI5: this is a small, deterministic sweep to prove tuning direction:
        # - extreme hunger should increase California outcomes
        # - sickness/exhaustion should increase Oregon outcomes
        engine = DecisionEngine()
        from wagon_train.simulation import build_default_party

        hunger_choices = {"oregon": 0, "california": 0, "split": 0}
        sickness_choices = {"oregon": 0, "california": 0, "split": 0}

        for seed in range(20):
            random.seed(seed)
            hungry_party = build_default_party()
            sick_party = build_default_party()

            hungry_world = WagonTrain()
            hungry_world.living_count = len(hungry_party)
            hungry_world.food_supply = len(hungry_party) * 1.5 * 1.2
            hungry_world.morale = 55.0

            sick_world = WagonTrain()
            sick_world.living_count = len(sick_party)
            sick_world.food_supply = len(sick_party) * 1.5 * 10.0
            sick_world.morale = 35.0
            sick_world.sickness_events = ["fever", "dysentery"]

            for agent in hungry_party:
                agent.hunger = 85.0
            for agent in sick_party:
                agent.health = 32.0
                agent.morale = 22.0

            hunger_route, _, _ = engine.resolve_trail_choice(hungry_party, hungry_world)
            sickness_route, _, _ = engine.resolve_trail_choice(sick_party, sick_world)

            hunger_choices[hunger_route] += 1
            sickness_choices[sickness_route] += 1

        assert hunger_choices["california"] > hunger_choices["oregon"]
        assert sickness_choices["oregon"] > sickness_choices["california"]

    def test_split_route_vote_follows_more_influential_party(self):
        engine = DecisionEngine()
        world = WagonTrain()
        agents = [
            Agent(
                "StrongSplit",
                Role.CAPTAIN,
                Traits(0.2, 0.5, 0.2, 0.9),
                influence=10.0,
            ),
            Agent(
                "CalScout",
                Role.SCOUT,
                Traits(0.8, 0.5, 0.5, 0.4),
                influence=2.0,
            ),
            Agent(
                "CalHunter",
                Role.HUNTER,
                Traits(0.8, 0.5, 0.5, 0.4),
                influence=2.0,
            ),
            Agent(
                "OregonMedic",
                Role.MEDIC,
                Traits(0.2, 0.5, 0.5, 0.4),
                influence=8.0,
            ),
        ]

        # Keep trust neutral for deterministic weighting.
        for a in agents:
            for b in agents:
                if a is not b:
                    a.relationships[b.name] = 0.5

        chosen_route, tally, message = engine.resolve_trail_choice(agents, world)

        assert tally["split"] > tally["oregon"]
        assert tally["split"] > tally["california"]
        assert chosen_route == "oregon"
        assert "split party" in message

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

    def test_crossing_safety_bonus_message_appears_on_ford(self, monkeypatch):
        # ELI5: if we have temporary crossing safety intel, ford messaging should
        # reflect reduced danger context.
        engine = DecisionEngine()
        world = WagonTrain(start_year=1855, start_month=4)
        world.river_ahead = True
        world.crossing_safety_days = 3
        party = [Agent("Scout", Role.SCOUT, Traits())]

        # Keep deterministic path through the method.
        monkeypatch.setattr(random, "random", lambda: 0.99)
        outcomes = engine.apply_action(Action.FORD_RIVER, party, world)

        assert any("reduce river danger" in msg for msg in outcomes)

    def test_hunt_may_add_food(self):
        random.seed(42)
        engine = DecisionEngine()
        world = WagonTrain()
        party = [Agent("Hunter", Role.HUNTER, Traits())]
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
        party = [Agent("Wheel", Role.WHEELWRIGHT, Traits())]
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

    def test_starvation_triggers_emergency_slaughter_and_refills_food(self):
        engine = DecisionEngine()
        world = WagonTrain(food_supply=0.0, oxen_count=1, horse_count=0, mule_count=0)
        party = [Agent("A", Role.PASSENGER, Traits())]

        msgs = engine._consume_food(party, world, Action.TRAVEL)

        assert any("Emergency slaughter" in m for m in msgs)
        assert world.oxen_count == 0
        assert world.food_supply > 0.0

    def test_travel_speed_drops_when_draft_animals_are_lost(self):
        engine = DecisionEngine()
        party = [Agent("A", Role.PASSENGER, Traits())]
        world_with_animals = WagonTrain(oxen_count=6, horse_count=2, mule_count=2)
        world_without_animals = WagonTrain(oxen_count=0, horse_count=0, mule_count=0)
        world_with_animals.weather = Weather.SUNNY
        world_without_animals.weather = Weather.SUNNY

        random.seed(123)
        msgs_full = engine._do_travel(party, world_with_animals, len(party))
        random.seed(123)
        msgs_reduced = engine._do_travel(party, world_without_animals, len(party))

        assert world_without_animals.miles_traveled < world_with_animals.miles_traveled
        assert any("travels" in m for m in msgs_full)
        assert any("travels" in m for m in msgs_reduced)

    def test_travel_speed_drops_in_mountain_segment_vs_plains(self):
        engine = DecisionEngine()
        party = [Agent("A", Role.PASSENGER, Traits())]

        # ELI5: same people, same weather, same wagon condition.
        # Only terrain position changes, so miles should be lower in mountains.
        plains_world = WagonTrain()
        plains_world.weather = Weather.SUNNY
        plains_world.miles_traveled = 100.0

        mountains_world = WagonTrain()
        mountains_world.weather = Weather.SUNNY
        mountains_world.miles_traveled = 1000.0

        random.seed(456)
        _ = engine._do_travel(party, plains_world, len(party))
        random.seed(456)
        msgs_mountain = engine._do_travel(party, mountains_world, len(party))

        plains_daily = plains_world._recent_miles[-1]
        mountain_daily = mountains_world._recent_miles[-1]
        assert mountain_daily < plains_daily
        assert any("terrain" in msg for msg in msgs_mountain)

    def test_repair_clears_urgent_repair_assignment(self):
        engine = DecisionEngine()
        world = WagonTrain()
        world.urgent_repair_assignee = "Wheel"
        party = [Agent("Wheel", Role.WHEELWRIGHT, Traits())]
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
        captain = Agent("Leader", Role.CAPTAIN, Traits(0.5, 0.7, 0.6, 0.8))
        follower = Agent("Follower", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5))
        # Follower highly trusts captain
        follower.relationships["Leader"] = 1.0
        party = [captain, follower]
        tally_high = engine.collect_votes(party, world)

        captain2 = Agent("Leader", Role.CAPTAIN, Traits(0.5, 0.7, 0.6, 0.8))
        follower2 = Agent("Follower", Role.PASSENGER, Traits(0.5, 0.5, 0.5, 0.5))
        # Follower distrusts captain2
        follower2.relationships["Leader"] = 0.0
        party2 = [captain2, follower2]
        tally_low = engine.collect_votes(party2, world)

        # Leader should have higher weighted vote when trusted
        captain_action_high = captain.propose_action(world)
        captain_action_low = captain2.propose_action(world)
        if captain_action_high == captain_action_low:
            assert tally_high.get(captain_action_high, 0) > tally_low.get(captain_action_low, 0)


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
            es.roll(world, party)
            # Just verify no crashes and values stay non-negative
            assert world.food_supply >= 0.0

    def test_negative_miles_event_never_moves_backwards(self):
        es = EventSystem()
        world = WagonTrain()
        world.miles_traveled = 100.0
        party = [Agent("A", Role.WHEELWRIGHT, Traits())]

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
            Agent("Wheelwright", Role.WHEELWRIGHT, Traits()),
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
        assert world.urgent_repair_assignee == "Wheelwright"

    def test_injured_animal_event_reduces_draft_animal_count(self):
        es = EventSystem()
        world = WagonTrain(oxen_count=1, horse_count=0, mule_count=0)
        party = [Agent("Passenger", Role.PASSENGER, Traits())]
        synthetic = {
            "name": "Synthetic Injured Animal",
            "description": "Testing draft animal attrition.",
            "food_delta": 0,
            "parts_delta": 0,
            "health_delta": 0,
            "morale_delta": 0,
            "sickness": False,
            "injure_draft_animal": True,
            # Force deterministic permanent loss for this test.
            "injure_draft_animal_loss_chance": 1.0,
        }
        es._choose_event = lambda: synthetic  # type: ignore[method-assign]
        es.BASE_EVENT_CHANCE = 1.0
        messages = es.roll(world, party)
        assert world.oxen_count == 0
        assert any("future travel speed is reduced" in msg for msg in messages)

    def test_warning_markers_event_sets_disease_warning_days(self):
        es = EventSystem()
        world = WagonTrain()
        party = [Agent("Passenger", Role.PASSENGER, Traits())]
        synthetic = {
            "name": "Trailside Warning Markers",
            "description": "Testing warning marker effect.",
            "food_delta": 0,
            "parts_delta": 0,
            "health_delta": 0,
            "morale_delta": 0,
            "sickness": False,
            "warning_markers": True,
        }
        es._choose_event = lambda: synthetic  # type: ignore[method-assign]
        es.BASE_EVENT_CHANCE = 1.0

        es.roll(world, party)
        assert world.disease_warning_days >= 6

    def test_disease_warning_reduces_sickness_event_impact(self):
        es = EventSystem()
        world = WagonTrain()
        world.disease_warning_days = 3
        party = [
            Agent("A", Role.PASSENGER, Traits(), health=100.0),
            Agent("B", Role.PASSENGER, Traits(), health=100.0),
        ]
        synthetic = {
            "name": "Dysentery Outbreak",
            "description": "Testing warning mitigation.",
            "food_delta": 0,
            "parts_delta": 0,
            "health_delta": -15,
            "morale_delta": -10,
            "sickness": True,
        }
        es._choose_event = lambda: synthetic  # type: ignore[method-assign]
        es.BASE_EVENT_CHANCE = 1.0

        es.roll(world, party)
        # Mitigated health loss should be 78% of 15 = 11.7, truncated to 11.
        assert all(a.health == pytest.approx(89.0) for a in party)

    def test_crossing_safety_event_sets_bonus_days(self):
        es = EventSystem()
        world = WagonTrain()
        party = [Agent("Passenger", Role.PASSENGER, Traits())]
        synthetic = {
            "name": "New Ferry and Bridge Crossing",
            "description": "Testing crossing safety intel.",
            "food_delta": 0,
            "parts_delta": 0,
            "health_delta": 0,
            "morale_delta": 0,
            "sickness": False,
            "crossing_safety": True,
        }
        es._choose_event = lambda: synthetic  # type: ignore[method-assign]
        es.BASE_EVENT_CHANCE = 1.0

        es.roll(world, party)
        assert world.crossing_safety_days >= 5


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
        assert Role.CAPTAIN in roles
        assert Role.HUNTER in roles
        assert Role.MEDIC in roles
        assert Role.WHEELWRIGHT in roles
        assert Role.SCOUT in roles
        assert Role.PASSENGER in roles
        assert Role.COOK in roles
        assert Role.BLACKSMITH in roles
        assert Role.GUARD in roles
        assert Role.HOSTLER in roles


    def test_simulation_collects_passenger_fares(self):
        sim = Simulation(seed=1, log_to_stdout=False)
        passenger_count = sum(1 for a in sim.agents if a.role == Role.PASSENGER)
        expected_cash = passenger_count * PASSENGER_FARE_DOLLARS
        assert sim.world.cash == pytest.approx(expected_cash)

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
