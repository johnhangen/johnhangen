import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from wingspan_rl.constants import N_ACTIONS  # noqa: E402
from wingspan_rl.env import WingspanEnv  # noqa: E402
from wingspan_rl.observation import OBS_SIZE  # noqa: E402


def rollout(env, seed=0, policy=None):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total, steps = 0.0, 0
    while True:
        legal = info["legal_actions"]
        action = policy(env, info) if policy else int(rng.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        assert obs.shape == (OBS_SIZE,)
        if terminated or truncated:
            return total, steps, info


def test_spaces():
    env = WingspanEnv(seed=0)
    assert env.action_space.n == N_ACTIONS
    assert env.observation_space.shape == (OBS_SIZE,)


def test_reset_returns_a_legal_masked_observation():
    env = WingspanEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert info["action_mask"].sum() == len(info["legal_actions"])
    assert np.array_equal(env.action_masks(), info["action_mask"])
    assert env.game.pending.player == env.agent_seat


def test_full_episode_terminates_with_scores():
    env = WingspanEnv(num_players=2, seed=1)
    _, steps, info = rollout(env, seed=1)
    assert steps > 20
    assert len(info["scores"]) == 2
    assert isinstance(info["is_win"], bool)
    assert sum(info["score_breakdown"].values()) > 0


@pytest.mark.parametrize("mode", ["dense", "sparse", "score", "score_diff"])
def test_reward_modes(mode):
    env = WingspanEnv(num_players=2, seed=2, reward_mode=mode)
    total, _, info = rollout(env, seed=2)
    assert np.isfinite(total)
    if mode == "sparse":
        assert total in (-1.0, 0.0, 1.0)
    if mode == "score":
        assert total == pytest.approx(info["scores"][0] / 100.0)


def test_illegal_action_raises_by_default():
    env = WingspanEnv(seed=3)
    _, info = env.reset(seed=3)
    illegal = next(a for a in range(N_ACTIONS) if a not in info["legal_actions"])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_illegal_action_can_be_penalized_instead():
    env = WingspanEnv(seed=3, illegal_action="penalize")
    _, info = env.reset(seed=3)
    illegal = next(a for a in range(N_ACTIONS) if a not in info["legal_actions"])
    obs, reward, terminated, truncated, info = env.step(illegal)
    assert reward == -1.0
    assert not terminated and not truncated
    assert info["illegal_action"] == illegal


def test_seeding_is_reproducible():
    def run(seed):
        env = WingspanEnv(num_players=2, seed=seed)
        return rollout(env, seed=seed)[0]

    assert run(7) == run(7)


def test_multi_player_seats_and_opponent_specs():
    env = WingspanEnv(num_players=4, agent_seat=2, opponents=["greedy", "random",
                                                              "greedy", "random"])
    obs, info = env.reset(seed=5)
    assert env.game.pending.player == 2
    _, _, info = rollout(env, seed=5)
    assert len(info["scores"]) == 4


def test_render_returns_text():
    env = WingspanEnv(seed=0, render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert "round" in text and "feeder" in text


def test_gymnasium_env_checker_passes_in_penalize_mode():
    from gymnasium.utils.env_checker import check_env

    check_env(WingspanEnv(num_players=2, seed=0, illegal_action="penalize"),
              skip_render_check=True)
