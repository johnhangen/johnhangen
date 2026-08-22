"""Train a masked PPO agent against the built-in bots.

Needs the training extras::

    pip install "stable-baselines3>=2.0" sb3-contrib torch

Then::

    python examples/train_maskable_ppo.py --timesteps 200000 --opponents greedy

The action space is large and mostly illegal at any given moment, so action
masking is not optional here - plain PPO wastes almost all of its samples.
"""

import argparse

from wingspan_rl.env import WingspanEnv


def mask_fn(env):
    # Monitor (and any other wrapper) does not forward action_masks.
    return env.unwrapped.action_masks()


def make_env(opponents: str, num_players: int, reward_mode: str, seed: int):
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    def _init():
        env = WingspanEnv(
            num_players=num_players,
            opponents=opponents,
            reward_mode=reward_mode,
            seed=seed,
        )
        return ActionMasker(Monitor(env), mask_fn)

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--opponents", default="greedy")
    parser.add_argument("--reward", default="dense")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--save", default="ppo_wingspan")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([
        make_env(args.opponents, args.players, args.reward, args.seed + i)
        for i in range(args.n_envs)
    ])
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        n_steps=256,
        batch_size=512,
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [512, 256]},
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)

    eval_env = make_env(args.opponents, args.players, "sparse", args.seed + 999)()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_games)
    print(f"sparse reward vs {args.opponents} "
          f"(+1 win / 0 tie / -1 loss): {mean_reward:.3f} +/- {std_reward:.3f}")


if __name__ == "__main__":
    main()
