"""Simulation loop for the wagon train multi-agent simulator."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .agent import Agent, Role, Traits
from .decisions import Action, DecisionEngine
from .events import EventSystem
from .logger import SimulationLogger
from .world import WagonTrain, Weather


# ---------------------------------------------------------------------------
# Default party roster
# ---------------------------------------------------------------------------

def build_default_party() -> List[Agent]:
    """Create the 20-agent default wagon roster.

    ELI5:
    - We keep a fixed ordered list so "first N" selection remains deterministic.
    - Main entrypoint now allows selecting 10-20 travelers from this roster.
    """
    party = [
        Agent("Eliza Hart",    Role.LEADER,    Traits(0.5, 0.7, 0.6, 0.8)),
        Agent("Tom Buckley",   Role.HUNTER,    Traits(0.8, 0.4, 0.5, 0.5)),
        Agent("Clara Quinn",   Role.MEDIC,     Traits(0.3, 0.9, 0.4, 0.9)),
        Agent("Amos Reed",     Role.MECHANIC,  Traits(0.4, 0.5, 0.7, 0.6)),
        Agent("Jesse Fox",     Role.SCOUT,     Traits(0.7, 0.4, 0.3, 0.5)),
        Agent("Mae Cooper",    Role.PASSENGER, Traits(0.4, 0.6, 0.5, 0.7)),
        Agent("Old Pete",      Role.PASSENGER, Traits(0.2, 0.5, 0.8, 0.4)),
        Agent("Rev. Silas Boone", Role.PREACHER, Traits(0.2, 0.8, 0.7, 0.9)),
        Agent("Ruby Dalton",   Role.PASSENGER, Traits(0.5, 0.7, 0.4, 0.8)),
        Agent("Caleb Stone",   Role.HUNTER,    Traits(0.7, 0.3, 0.6, 0.4)),
        Agent("Nora Pike",     Role.PASSENGER, Traits(0.4, 0.6, 0.5, 0.6)),
        Agent("Elias Ward",    Role.MECHANIC,  Traits(0.4, 0.5, 0.6, 0.5)),
        Agent("Hattie Sloan",  Role.MEDIC,     Traits(0.3, 0.9, 0.4, 0.8)),
        Agent("Micah Graves",  Role.SCOUT,     Traits(0.7, 0.4, 0.3, 0.5)),
        Agent("June Holloway", Role.PASSENGER, Traits(0.5, 0.7, 0.5, 0.7)),
        Agent("Wyatt Dunn",    Role.HUNTER,    Traits(0.8, 0.3, 0.5, 0.4)),
        Agent("Abigail Frost", Role.PASSENGER, Traits(0.4, 0.7, 0.5, 0.7)),
        Agent("Jonah Webb",    Role.PASSENGER, Traits(0.4, 0.6, 0.6, 0.6)),
        Agent("Martha Bell",   Role.PREACHER,  Traits(0.2, 0.8, 0.7, 0.9)),
        Agent("Gideon Marsh",  Role.PASSENGER, Traits(0.4, 0.5, 0.6, 0.6)),
    ]
    return party


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------

class Simulation:
    """Runs the full wagon train simulation."""

    def __init__(
        self,
        agents: Optional[List[Agent]] = None,
        world: Optional[WagonTrain] = None,
        seed: Optional[int] = None,
        log_to_stdout: bool = True,
    ) -> None:
        if seed is not None:
            random.seed(seed)

        self.agents: List[Agent] = agents if agents is not None else build_default_party()
        self.world: WagonTrain = world if world is not None else WagonTrain()
        self.engine = DecisionEngine()
        self.events = EventSystem()
        self.logger = SimulationLogger(log_to_stdout=log_to_stdout)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Execute the simulation and return the end-reason string."""
        while not self.world.is_finished:
            living = [a for a in self.agents if a.alive]
            if not living:
                reason = "All travelers have perished."
                self.logger.log_simulation_end(self.world, self.agents, reason)
                return reason

            # 1. Advance day (weather, spoilage, wear)
            self.world.advance_day()
            # Capture where the day starts so we can measure net daily progress.
            day_start_miles = self.world.miles_traveled

            # 2. Log day header
            self.logger.log_day_header(self.world)

            # 3. Random events
            event_messages = self.events.roll(self.world, self.agents)
            self.logger.log_events(event_messages)

            # 4. Collect proposals & resolve vote
            proposals: Dict[str, Action] = {}
            for agent in living:
                proposals[agent.name] = agent.propose_action(self.world)

            winning_action, tally = self.engine.resolve(living, self.world)

            # 5. Log proposals and vote
            self.logger.log_agent_proposals(self.agents, proposals)
            self.logger.log_votes(tally, winning_action)

            # 6. Apply winning action
            outcomes = self.engine.apply_action(winning_action, self.agents, self.world)
            self.logger.log_outcomes(outcomes)

            # 8. Record non-negative net progress for the entire day
            # (events + chosen action combined).
            self.world.record_daily_progress(self.world.miles_traveled - day_start_miles)

            # 9. Agent status
            self.logger.log_agent_status(self.agents)

        # Determine end reason
        if self.world.miles_traveled >= self.world.GOAL_MILES:
            reason = (
                f"Oregon reached after {self.world.day} days! "
                f"({sum(1 for a in self.agents if a.alive)} survivors)"
            )
        else:
            reason = f"200-day limit reached. Miles covered: {self.world.miles_traveled:.1f}"

        self.logger.log_simulation_end(self.world, self.agents, reason)
        return reason
