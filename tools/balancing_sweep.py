#!/usr/bin/env python3
"""Run a multi-seed balancing sweep and emit CSV trend metrics.

ELI5 summary:
- We run the game many times with different random seeds.
- For each run, we capture "how it ended" and "what resources were left".
- We also compute running trend lines so balancing changes are easy to compare.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
import sys
from pathlib import Path

# Ensure project root is importable when script is run as `python tools/...`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wagon_train.simulation import Simulation, build_default_party
from wagon_train.world import TRAIL_DESTINATION


@dataclass(frozen=True)
class SweepRow:
    """One simulation result row plus running-trend columns.

    ELI5:
    - Think of this as one spreadsheet row.
    - The first columns are raw facts from one simulation.
    - The later columns are trend numbers accumulated up to that row.
    """

    seed: int
    reached_destination: bool
    end_reason: str
    day_to_finish: int
    miles_traveled: float
    end_food: float
    end_parts: float
    end_morale: float
    survivors: int
    # Running trend columns (cumulative through this seed index).
    reach_rate_to_date: float
    avg_day_to_finish_reached_to_date: float
    avg_end_food_to_date: float
    avg_end_parts_to_date: float
    avg_end_morale_to_date: float


# ELI5: writing a tiny helper function keeps the main loop easier to read.
def run_single_seed(seed: int, party_size: int) -> tuple[bool, str, int, float, float, float, float, int]:
    """Run a single seed and return the core balancing metrics tuple."""
    party = build_default_party()[:party_size]
    sim = Simulation(agents=party, seed=seed, log_to_stdout=False)
    reason = sim.run()

    reached = sim.world.miles_traveled >= sim.world.GOAL_MILES
    survivors = sum(1 for agent in sim.agents if agent.alive)

    return (
        reached,
        reason,
        sim.world.day,
        sim.world.miles_traveled,
        sim.world.food_supply,
        sim.world.wagon_parts,
        sim.world.morale,
        survivors,
    )


# ELI5: this function turns a list of seeds into rows with trend lines.
def build_rows(start_seed: int, runs: int, party_size: int) -> list[SweepRow]:
    """Execute the sweep and return fully computed report rows."""
    rows: list[SweepRow] = []

    reached_count = 0
    reached_day_total = 0
    reached_day_runs = 0
    end_food_total = 0.0
    end_parts_total = 0.0
    end_morale_total = 0.0

    for index in range(runs):
        seed = start_seed + index
        (
            reached,
            reason,
            day_to_finish,
            miles_traveled,
            end_food,
            end_parts,
            end_morale,
            survivors,
        ) = run_single_seed(seed=seed, party_size=party_size)

        if reached:
            reached_count += 1
            reached_day_total += day_to_finish
            reached_day_runs += 1

        end_food_total += end_food
        end_parts_total += end_parts
        end_morale_total += end_morale

        completed_runs = index + 1
        reach_rate_to_date = reached_count / completed_runs
        # ELI5: if nobody has reached the destination yet, there is no average
        # day-to-finish for winners, so we use 0.0 as a placeholder.
        avg_day_to_finish_reached = (
            reached_day_total / reached_day_runs if reached_day_runs else 0.0
        )

        rows.append(
            SweepRow(
                seed=seed,
                reached_destination=reached,
                end_reason=reason,
                day_to_finish=day_to_finish,
                miles_traveled=miles_traveled,
                end_food=end_food,
                end_parts=end_parts,
                end_morale=end_morale,
                survivors=survivors,
                reach_rate_to_date=reach_rate_to_date,
                avg_day_to_finish_reached_to_date=avg_day_to_finish_reached,
                avg_end_food_to_date=end_food_total / completed_runs,
                avg_end_parts_to_date=end_parts_total / completed_runs,
                avg_end_morale_to_date=end_morale_total / completed_runs,
            )
        )

    return rows


# ELI5: this is where we physically save the CSV spreadsheet file.
def write_csv(rows: list[SweepRow], output_csv: Path) -> None:
    """Write sweep rows to disk as CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(SweepRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


# ELI5: this prints a quick summary so you don't need to open the CSV first.
def print_summary(rows: list[SweepRow]) -> None:
    """Print concise run summary to stdout."""
    if not rows:
        print("No rows produced.")
        return

    final = rows[-1]
    reached_total = sum(1 for row in rows if row.reached_destination)
    print("Balancing sweep complete")
    print(f"Runs: {len(rows)}")
    print(f"Reached {TRAIL_DESTINATION}: {reached_total}/{len(rows)} ({final.reach_rate_to_date:.1%})")
    print(f"Avg day-to-finish (successful runs): {final.avg_day_to_finish_reached_to_date:.2f}")
    print(
        "Avg end resources: "
        f"food={final.avg_end_food_to_date:.2f}, "
        f"parts={final.avg_end_parts_to_date:.2f}, "
        f"morale={final.avg_end_morale_to_date:.2f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args for balancing sweeps."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an extended balancing sweep over sequential seeds and emit "
            "a CSV with trend-line columns."
        )
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="How many sequential seeds to run (default: 30).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="First seed to run (default: 1).",
    )
    parser.add_argument(
        "--party-size",
        type=int,
        default=20,
        help="Party size to simulate (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/reports/balancing_sweep.csv"),
        help="Where to save the CSV report.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if args.runs <= 0:
        raise SystemExit("--runs must be > 0")
    if not (10 <= args.party_size <= len(build_default_party())):
        raise SystemExit("--party-size must be between 10 and 20")

    rows = build_rows(start_seed=args.start_seed, runs=args.runs, party_size=args.party_size)
    write_csv(rows=rows, output_csv=args.output)
    print_summary(rows)
    print(f"CSV written: {args.output}")


if __name__ == "__main__":
    main()
