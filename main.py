#!/usr/bin/env python3
"""Entry point for the multi-agent Oregon Trail wagon train simulator.

Usage:
    python main.py [--seed SEED] [--no-log]
"""

import argparse
import sys

from wagon_train.simulation import Simulation


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
    args = parser.parse_args()

    sim = Simulation(seed=args.seed, log_to_stdout=not args.no_log)
    reason = sim.run()

    if args.no_log:
        print(f"\nSimulation complete: {reason}")
        print(f"Days elapsed : {sim.world.day}")
        print(f"Miles covered: {sim.world.miles_traveled:.1f}")
        survivors = [a for a in sim.agents if a.alive]
        print(f"Survivors    : {len(survivors)} — {', '.join(a.name for a in survivors)}")


if __name__ == "__main__":
    main()
