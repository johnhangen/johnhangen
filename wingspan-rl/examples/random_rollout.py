"""Play one game through the raw engine API and print the result.

    python examples/random_rollout.py --seed 3
"""

import argparse
import random

from wingspan_rl import GameConfig, WingspanGame, render_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-decisions", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    game = WingspanGame(GameConfig(num_players=args.players, seed=args.seed))

    while not game.done:
        decision = game.pending
        action = rng.choice(decision.action_ids())
        if args.show_decisions:
            print(f"P{decision.player} {decision.prompt} -> "
                  f"{decision.by_action(action).label}")
        game.step(action)

    print(render_text(game))


if __name__ == "__main__":
    main()
