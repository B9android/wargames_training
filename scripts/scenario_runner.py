"""Headless scenario runner for quick integration checks.

Sets up a deterministic 1v1 battalion scenario, runs it for a fixed number
of steps, and asserts key simulation invariants hold. Suitable for CI and
local smoke-testing without requiring a display.

Usage:
    python scripts/scenario_runner.py [--steps N] [--seed S]

Exit codes:
    0 — all integration checks passed
    1 — one or more checks failed
"""

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.battalion import Battalion


def build_scenario() -> tuple[Battalion, Battalion]:
    """Create a deterministic 1v1 scenario: blue faces red within fire range."""
    # Blue at (300, 500) facing right; red at (450, 500) facing left.
    # Separation (150 m) is deliberately inside the 200 m fire_range so both
    # sides can fire immediately without any movement phase.
    blue = Battalion(x=300.0, y=500.0, theta=0.0, strength=1.0, team=0)
    red = Battalion(x=450.0, y=500.0, theta=math.pi, strength=1.0, team=1)
    return blue, red


def step(blue: Battalion, red: Battalion) -> float:
    """Advance one simulation step. Returns total damage dealt this step."""
    damage = 0.0
    damage += blue.fire_at(red, intensity=1.0)
    damage += red.fire_at(blue, intensity=1.0)
    return damage


def run_scenario(steps: int = 200, seed: int = 42) -> dict:
    """Run the full scenario and return a summary metrics dict."""
    blue, red = build_scenario()

    initial_blue = blue.strength
    initial_red = red.strength
    total_damage = 0.0

    for _ in range(steps):
        total_damage += step(blue, red)

    return {
        "steps": steps,
        "seed": seed,
        "blue_strength_initial": initial_blue,
        "blue_strength_final": blue.strength,
        "red_strength_initial": initial_red,
        "red_strength_final": red.strength,
        "total_damage": total_damage,
        "blue": blue,
        "red": red,
    }


def check_results(results: dict) -> list[str]:
    """Return a list of failed check descriptions.

    An empty list means all integration checks passed.
    """
    failures: list[str] = []

    blue_final = results["blue_strength_final"]
    red_final = results["red_strength_final"]

    # Strength must remain in valid range [0, 1].
    if not (0.0 <= blue_final <= 1.0):
        failures.append(f"Blue strength {blue_final:.4f} out of range [0, 1]")
    if not (0.0 <= red_final <= 1.0):
        failures.append(f"Red strength {red_final:.4f} out of range [0, 1]")

    # Both sides must have taken damage given the symmetric in-range setup.
    if blue_final >= results["blue_strength_initial"]:
        failures.append("Blue took no damage despite being in red's fire arc")
    if red_final >= results["red_strength_initial"]:
        failures.append("Red took no damage despite being in blue's fire arc")

    # Total damage must be positive.
    if results["total_damage"] <= 0.0:
        failures.append("No damage was dealt over the scenario (expected > 0)")

    # Positions must remain finite — guards against numerical blow-ups.
    for name, battalion in [("Blue", results["blue"]), ("Red", results["red"])]:
        if not math.isfinite(battalion.x) or not math.isfinite(battalion.y):
            failures.append(f"{name} position contains non-finite value")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Headless scenario runner for quick integration checks."
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of simulation steps (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args(argv)

    print(f"Running scenario: steps={args.steps}, seed={args.seed}")
    results = run_scenario(steps=args.steps, seed=args.seed)

    print(f"  Blue strength : {results['blue_strength_initial']:.3f} → {results['blue_strength_final']:.3f}")
    print(f"  Red  strength : {results['red_strength_initial']:.3f} → {results['red_strength_final']:.3f}")
    print(f"  Total damage  : {results['total_damage']:.4f}")

    failures = check_results(results)
    if failures:
        print(f"\nFAIL — {len(failures)} check(s) failed:")
        for failure in failures:
            print(f"  ✗ {failure}")
        return 1

    print(f"\nPASS — all {results['steps']} steps completed, integration checks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
