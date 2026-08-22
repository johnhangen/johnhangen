"""Self-play skeleton over the multi-agent AEC environment.

Every seat is driven by the same callable, so dropping in a learned policy is
a one-line change.  Run::

    python examples/selfplay_aec.py --games 20
"""

import argparse
import random
import statistics

from wingspan_rl.aec_env import WingspanAECEnv
from wingspan_rl.agents import make_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--policy", default="greedy", choices=["greedy", "random"])
    args = parser.parse_args()

    env = WingspanAECEnv(num_players=args.players, reward_mode="sparse")
    policy = make_agent(args.policy, seed=0)
    scores = {a: [] for a in env.possible_agents}
    wins = {a: 0 for a in env.possible_agents}

    for game_index in range(args.games):
        env.reset(seed=game_index)
        for agent in env.agent_iter():
            _, _, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                if not env.agents:
                    break
                continue
            # The engine object is available for scripted policies; a neural
            # policy would use env.observe(agent)["observation"] instead.
            env.step(policy.act(env.game, env.seat(agent)))
        totals = info["scores"]
        for agent in env.possible_agents:
            scores[agent].append(totals[env.seat(agent)])
        for seat in info["winners"]:
            wins[env.possible_agents[seat]] += 1

    for agent in env.possible_agents:
        print(f"{agent}: wins {wins[agent]:>3}  "
              f"mean score {statistics.mean(scores[agent]):6.2f}")


if __name__ == "__main__":
    main()
