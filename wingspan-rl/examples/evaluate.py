"""Evaluate a trained MaskablePPO policy against the scripted baselines.

    python examples/evaluate.py --model ppo_wingspan --games 100
"""

import argparse
import statistics

from wingspan_rl.env import WingspanEnv


def evaluate(model, opponent: str, games: int, players: int, seed: int):
    env = WingspanEnv(num_players=players, opponents=opponent,
                      reward_mode="sparse", seed=seed)
    wins = ties = 0
    scores = []
    for game_index in range(games):
        obs, info = env.reset(seed=seed + game_index)
        while True:
            action, _ = model.predict(
                obs, action_masks=info["action_mask"], deterministic=True
            )
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        scores.append(info["scores"][env.agent_seat])
        if info["is_win"]:
            wins += 1
            ties += len(info["winners"]) > 1
    return wins, ties, statistics.mean(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_wingspan")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(args.model)
    for opponent in ("random", "greedy"):
        wins, ties, mean_score = evaluate(model, opponent, args.games,
                                          args.players, args.seed)
        print(f"vs {opponent:>7}: {wins}/{args.games} wins "
              f"({ties} shared), mean score {mean_score:.2f}")


if __name__ == "__main__":
    main()
