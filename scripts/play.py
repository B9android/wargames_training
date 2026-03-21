#!/usr/bin/env python3
# scripts/play.py
"""Interactive human-vs-AI game runner for BattalionEnv.

Provides a simple CLI lobby for selecting a scenario and difficulty,
then runs the Pygame game loop so a human can play against the AI.

Usage::

    python scripts/play.py                           # interactive lobby
    python scripts/play.py --scenario open_field     # skip lobby prompt
    python scripts/play.py --scenario mountain_pass --difficulty 3
    python scripts/play.py --scenario last_stand --policy checkpoints/policy.zip
    python scripts/play.py --list-scenarios          # print available scenarios

Controls during play::

    W / Arrow Up    — move forward
    S / Arrow Down  — move backward
    A / Arrow Left  — rotate left (counter-clockwise)
    D / Arrow Right — rotate right (clockwise)
    Space           — fire
    Escape          — quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on the path when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from envs.human_env import SCENARIOS, HumanEnv

# ---------------------------------------------------------------------------
# Lobby helpers
# ---------------------------------------------------------------------------

_CONTROLS_HELP = (
    "Controls: W/S/↑↓ move · A/D/←→ rotate · SPACE fire · ESC quit"
)


def _list_scenarios() -> None:
    """Print the available scenarios to stdout."""
    print("\nAvailable scenarios:\n")
    for name, cfg in SCENARIOS.items():
        desc = cfg.get("description", "")
        lvl = cfg.get("curriculum_level", "?")
        init_str = cfg.get("initial_blue_strength", 1.0)
        extras = []
        if cfg.get("randomize_terrain"):
            extras.append("random terrain")
        if float(cfg.get("hill_speed_factor", 0.5)) < 0.5:
            extras.append("steep hills")
        if float(init_str) < 1.0:
            extras.append(f"Blue starts at {int(float(init_str) * 100)}% strength")
        extras_str = f"  [{', '.join(extras)}]" if extras else ""
        print(f"  {name:<20} (difficulty {lvl}) — {desc}{extras_str}")
    print()


def _prompt_scenario() -> str:
    """Interactive CLI prompt: ask the player to choose a scenario."""
    scenario_list = list(SCENARIOS)
    print("\n=== Wargames Training — Human vs AI ===\n")
    print("Select a scenario:\n")
    for i, name in enumerate(scenario_list, start=1):
        desc = SCENARIOS[name].get("description", "")
        print(f"  [{i}] {name:<20} {desc}")
    print()
    while True:
        raw = input("Enter number or name (default: 1): ").strip()
        if not raw:
            return scenario_list[0]
        if raw.isdigit() and 1 <= int(raw) <= len(scenario_list):
            return scenario_list[int(raw) - 1]
        if raw in SCENARIOS:
            return raw
        print(f"  Invalid choice {raw!r}. Please try again.")


def _prompt_difficulty() -> int:
    """Interactive CLI prompt: ask the player to choose a difficulty level."""
    levels = [
        (1, "Stationary — Red does not move or fire"),
        (2, "Turning only — Red faces you but stays put"),
        (3, "Advancing — Red moves toward you; no fire"),
        (4, "Combat — Red moves and fires at 50% intensity"),
        (5, "Full combat — Red moves and fires at 100% intensity (default)"),
    ]
    print("\nSelect AI difficulty:\n")
    for lvl, desc in levels:
        print(f"  [{lvl}] {desc}")
    print()
    while True:
        raw = input("Enter level 1–5 (default: 5): ").strip()
        if not raw:
            return 5
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  Please enter a number between 1 and 5.")


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------


def _load_policy(policy_path: str):
    """Load an SB3 policy from *policy_path*.

    Returns the loaded model on success, or ``None`` when loading fails
    (with a warning printed to stderr).
    """
    path = Path(policy_path)
    if not path.exists():
        print(
            f"Warning: policy checkpoint not found: {policy_path!r}. "
            "Falling back to scripted opponent.",
            file=sys.stderr,
        )
        return None
    try:
        from stable_baselines3 import PPO  # noqa: PLC0415

        model = PPO.load(str(path))
        print(f"Loaded AI policy: {path.name}")
        return model
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: could not load policy ({exc}). "
            "Falling back to scripted opponent.",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def run_game(
    scenario: str,
    difficulty: Optional[int] = None,
    red_policy=None,
    seed: Optional[int] = None,
) -> dict:
    """Run a single game episode and return the result.

    Parameters
    ----------
    scenario:
        Name of the scenario (key in :data:`~envs.human_env.SCENARIOS`).
    difficulty:
        AI curriculum level (1–5).  When ``None``, the scenario's own
        ``curriculum_level`` is used unchanged.
    red_policy:
        Optional loaded SB3 model to use as the AI opponent.
    seed:
        Random seed for reproducible episode layout.

    Returns
    -------
    dict
        Result dict with keys ``winner`` (str), ``steps`` (int), and
        ``total_reward`` (float).
    """
    env = HumanEnv.from_scenario(scenario, difficulty=difficulty, red_policy=red_policy)
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    result: dict = {"winner": "unknown", "steps": 0, "total_reward": 0.0}

    print(f"\nScenario : {env.scenario_name}")
    print(f"          {env.scenario_description}")
    print(_CONTROLS_HELP)
    print()

    try:
        while True:
            action, quit_req = env.poll_action()
            if quit_req:
                result = {
                    "winner": "quit",
                    "steps": env.step_count,
                    "total_reward": total_reward,
                }
                break

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()

            if terminated or truncated:
                result = {
                    "winner": info.get("winner", "unknown"),
                    "steps": env.step_count,
                    "total_reward": total_reward,
                }
                break
    finally:
        env.close()

    return result


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------


def print_result(result: dict) -> None:
    """Print the end-of-game result to stdout."""
    winner = result.get("winner", "unknown")
    steps = result.get("steps", 0)
    total_reward = result.get("total_reward", 0.0)

    print("\n" + "=" * 42)
    if winner == "blue":
        print("  VICTORY — You defeated the enemy!")
    elif winner == "red":
        print("  DEFEAT  — Your battalion was routed.")
    elif winner == "draw":
        print("  DRAW    — Both sides exhausted.")
    elif winner == "quit":
        print("  Game quit by player.")
    else:
        print(f"  Result  : {winner}")
    print(f"  Steps   : {steps}")
    print(f"  Reward  : {total_reward:.2f}")
    print("=" * 42 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Parse command-line arguments and run the game."""
    parser = argparse.ArgumentParser(
        prog="play",
        description="Human vs AI — Wargames Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
        default=None,
        metavar="NAME",
        help="Scenario name (default: interactive selection).",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, 6),
        default=None,
        metavar="LEVEL",
        help="AI difficulty 1–5 (default: interactive selection or scenario default).",
    )
    parser.add_argument(
        "--policy",
        default=None,
        metavar="PATH",
        help="Path to a trained SB3 .zip checkpoint for the AI opponent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducible episode layout.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_scenarios:
        _list_scenarios()
        return

    # Resolve scenario and difficulty.
    scenario = args.scenario if args.scenario is not None else _prompt_scenario()

    if args.difficulty is not None:
        # Explicit CLI flag always wins.
        difficulty: Optional[int] = args.difficulty
    elif args.scenario is not None:
        # Scenario given on CLI without --difficulty: use the scenario's own
        # curriculum_level so the user is not prompted unnecessarily.
        difficulty = SCENARIOS[scenario].get("curriculum_level", 5)
    else:
        # Fully interactive lobby: ask the player.
        difficulty = _prompt_difficulty()
    red_policy = _load_policy(args.policy) if args.policy else None

    result = run_game(
        scenario=scenario,
        difficulty=difficulty,
        red_policy=red_policy,
        seed=args.seed,
    )
    print_result(result)


if __name__ == "__main__":
    main()
