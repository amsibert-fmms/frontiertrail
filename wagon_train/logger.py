"""Logging system for the wagon train simulator."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .agent import Agent
    from .decisions import Action
    from .world import WagonTrain


def _build_logger(name: str = "wagon_train") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class SimulationLogger:
    """Formats and emits readable per-day log entries."""

    SEPARATOR = "=" * 72

    def __init__(self, log_to_stdout: bool = True) -> None:
        self._logger = _build_logger() if log_to_stdout else None
        self.history: List[str] = []   # full text log for inspection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_day_header(
        self,
        world: "WagonTrain",
    ) -> None:
        lines = [
            self.SEPARATOR,
            world.summary(),
        ]
        self._emit(lines)

    def log_votes(
        self,
        tally: Dict["Action", float],
        winner: "Action",
    ) -> None:
        lines = ["  Votes:"]
        sorted_votes = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        for action, weight in sorted_votes:
            marker = " ◀ CHOSEN" if action == winner else ""
            lines.append(f"    {action.value:<14} {weight:>5.2f} influence{marker}")
        self._emit(lines)

    def log_agent_proposals(
        self,
        agents: List["Agent"],
        proposals: Dict[str, "Action"],
    ) -> None:
        lines = ["  Agent proposals:"]
        for agent in agents:
            if not agent.alive:
                lines.append(f"    {agent.name:<14} (deceased)")
                continue
            action = proposals.get(agent.name, "—")
            action_str = action.value if hasattr(action, "value") else str(action)
            lines.append(
                f"    {agent.name:<14} [{agent.role.value:<9}] "
                f"→ {action_str:<14} "
                f"(infl={agent.effective_influence:.2f}, "
                f"hp={agent.health:>4.0f}, "
                f"morale={agent.morale:>4.0f})"
            )
        self._emit(lines)

    def log_events(self, event_messages: List[str]) -> None:
        if event_messages:
            self._emit(event_messages)

    def log_outcomes(self, outcome_messages: List[str]) -> None:
        for msg in outcome_messages:
            self._emit([f"  → {msg}"])

    def log_agent_status(self, agents: List["Agent"]) -> None:
        lines = ["  Party status:"]
        for agent in agents:
            if not agent.alive:
                lines.append(f"    ✝ {agent.name:<14} (deceased)")
                continue
            lines.append(
                f"    {agent.name:<14} hp={agent.health:>4.0f}  "
                f"hunger={agent.hunger:>4.0f}  morale={agent.morale:>4.0f}"
            )
        self._emit(lines)

    def log_simulation_end(
        self,
        world: "WagonTrain",
        agents: List["Agent"],
        reason: str,
    ) -> None:
        living = [a for a in agents if a.alive]
        lines = [
            self.SEPARATOR,
            f"SIMULATION ENDED — {reason}",
            f"  Final day   : {world.day}",
            f"  Miles       : {world.miles_traveled:.1f} / {world.GOAL_MILES}",
            f"  Survivors   : {len(living)} / {len(agents)}",
        ]
        if living:
            lines.append("  Names       : " + ", ".join(a.name for a in living))
        self._emit(lines)
        self._emit([self.SEPARATOR])

    # ------------------------------------------------------------------
    # Internal emit
    # ------------------------------------------------------------------

    def _emit(self, lines: List[str]) -> None:
        text = "\n".join(lines)
        self.history.append(text)
        if self._logger:
            for line in lines:
                self._logger.info(line)
