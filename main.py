#!/usr/bin/env python3
"""Entry point for the multi-agent Oregon Trail wagon train simulator.

Usage:
    python main.py [--seed SEED] [--no-log]
"""

import argparse
import sys

from wagon_train.simulation import Simulation, build_default_party


def _prompt_party_size(default_size: int) -> int:
    """Ask the user how many people should be in the party.

    This prompt is intentionally resilient:
    - blank input => default
    - invalid input => re-prompt
    - EOF/non-interactive => default
    """
    while True:
        try:
            raw = input(
                f"How many people should be in the party? [1-{default_size}] "
                f"(Enter for {default_size}): "
            ).strip()
        except EOFError:
            return default_size

        if raw == "":
            return default_size

        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= default_size:
                return value

        print(f"Please enter a whole number from 1 to {default_size}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent Oregon Trail wagon train simulator"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Suppress per-day log output (only print end summary)",
    )
    parser.add_argument(
        "--party-size",
        type=int,
        default=None,
        help="Party size to use (1-10). If omitted, you will be prompted.",
    )
    args = parser.parse_args()

    default_party = build_default_party()
    max_party_size = len(default_party)

    if args.party_size is None:
        party_size = _prompt_party_size(max_party_size)
    else:
        if not (1 <= args.party_size <= max_party_size):
            parser.error(f"--party-size must be between 1 and {max_party_size}")
        party_size = args.party_size

    # Keep role diversity from the curated roster by taking the first N members.
    selected_party = default_party[:party_size]

    sim = Simulation(agents=selected_party, seed=args.seed, log_to_stdout=not args.no_log)
    reason = sim.run()

    if args.no_log:
        print(f"\nSimulation complete: {reason}")
        print(f"Days elapsed : {sim.world.day}")
        print(f"Miles covered: {sim.world.miles_traveled:.1f}")
        survivors = [a for a in sim.agents if a.alive]
        print(f"Survivors    : {len(survivors)} — {', '.join(a.name for a in survivors)}")


if __name__ == "__main__":
    main()
