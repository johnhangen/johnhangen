"""Gymnasium environment: one learning agent seated against scripted bots."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised by the import itself
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for wingspan_rl.env; install with `pip install gymnasium`"
    ) from exc

from .agents import make_agent
from .constants import N_ACTIONS
from .engine import GameConfig, WingspanGame
from .observation import OBS_SIZE, ObservationEncoder
from .render import render_text
from .scoring import score_player

REWARD_MODES = ("dense", "sparse", "score", "score_diff")


class WingspanEnv(gym.Env):
    """Single-agent Wingspan.

    The agent occupies ``agent_seat``; every other seat is played by a scripted
    opponent.  Actions are a flat :class:`~gymnasium.spaces.Discrete` space and
    only the ids in ``info["action_mask"]`` are legal at any moment - the mask
    is also available through :meth:`action_masks` for sb3-contrib's MaskablePPO.

    Reward modes:
        ``dense``      change in the agent's own score each step, plus the
                       final margin over the best opponent (default)
        ``sparse``     +1 win / 0 tie / -1 loss, at the end only
        ``score``      final score / 100, at the end only
        ``score_diff`` final margin over the best opponent / 50, at the end only
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "wingspan_v0"}

    def __init__(
        self,
        num_players: int = 2,
        opponents: Any = "greedy",
        agent_seat: int = 0,
        reward_mode: str = "dense",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        deck_path: Optional[str] = None,
        illegal_action: str = "raise",
        **game_options: Any,
    ):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if illegal_action not in ("raise", "penalize"):
            raise ValueError("illegal_action must be 'raise' or 'penalize'")
        if not 0 <= agent_seat < num_players:
            raise ValueError("agent_seat must be a valid seat index")

        self.num_players = num_players
        self.agent_seat = agent_seat
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.illegal_action = illegal_action
        self.deck_path = deck_path
        self.game_options = game_options
        self._opponent_spec = opponents
        self._seed = seed
        self._np_random_seed = seed

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )

        self.game: Optional[WingspanGame] = None
        self.encoder: Optional[ObservationEncoder] = None
        self._opponents: Dict[int, Any] = {}
        self._last_score = 0.0

    # ------------------------------------------------------------------ API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        config = GameConfig(
            num_players=self.num_players,
            seed=seed,
            deck_path=self.deck_path,
            **self.game_options,
        )
        self.game = WingspanGame(config)
        if self.encoder is None:
            self.encoder = ObservationEncoder(self.game.cards)
        specs = self._opponent_spec
        if not isinstance(specs, (list, tuple)):
            specs = [specs] * self.num_players
        self._opponents = {
            seat: make_agent(specs[seat % len(specs)], seed + 1000 + seat)
            for seat in range(self.num_players)
            if seat != self.agent_seat
        }
        self._last_score = self._own_score()
        self._run_opponents()
        return self._obs(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.game is not None, "call reset() first"
        action = int(action)
        legal = self.game.legal_action_ids()
        if self.game.done:
            raise RuntimeError("step() called on a finished game; call reset()")
        if self.game.pending.player != self.agent_seat:  # pragma: no cover - guarded
            raise RuntimeError("it is not the agent's turn")
        if action not in legal:
            if self.illegal_action == "raise":
                raise ValueError(
                    f"illegal action {action}; legal actions are {legal} "
                    f"({self.game.pending.prompt})"
                )
            obs, info = self._obs(), self._info()
            info["illegal_action"] = action
            return obs, -1.0, False, False, info

        self.game.step(action)
        self._run_opponents()

        reward = self._reward()
        terminated = self.game.done
        info = self._info()
        if terminated:
            scores = [s.total for s in self.game.scores()]
            info["scores"] = scores
            info["winners"] = self.game.winners()
            info["is_win"] = self.agent_seat in info["winners"]
            info["score_breakdown"] = self.game.scores()[self.agent_seat].as_dict()
        return self._obs(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO hook."""
        assert self.game is not None
        return self.game.action_mask()

    def render(self):
        assert self.game is not None
        text = render_text(self.game, self.agent_seat)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    # -------------------------------------------------------------- internals
    def _run_opponents(self) -> None:
        game = self.game
        assert game is not None
        guard = 0
        while not game.done and game.pending.player != self.agent_seat:
            seat = game.pending.player
            game.step(self._opponents[seat].act(game, seat))
            guard += 1
            if guard > 100_000:  # pragma: no cover - safety net
                raise RuntimeError("opponents failed to yield control")

    def _own_score(self) -> float:
        game = self.game
        assert game is not None
        board = game.state.board(self.agent_seat)
        return float(score_player(board, game.cards, game.bonus_cards).total)

    def _reward(self) -> float:
        game = self.game
        assert game is not None
        if self.reward_mode == "dense":
            score = self._own_score()
            reward = (score - self._last_score) / 10.0
            self._last_score = score
            if game.done:
                reward += self._final_margin() / 20.0
            return reward
        if not game.done:
            return 0.0
        if self.reward_mode == "sparse":
            winners = game.winners()
            if self.agent_seat in winners:
                return 1.0 if len(winners) == 1 else 0.0
            return -1.0
        if self.reward_mode == "score":
            return self._own_score() / 100.0
        return self._final_margin() / 50.0

    def _final_margin(self) -> float:
        game = self.game
        assert game is not None
        totals = [s.total for s in game.scores()]
        mine = totals[self.agent_seat]
        others = [t for i, t in enumerate(totals) if i != self.agent_seat]
        return float(mine - max(others)) if others else float(mine)

    def _obs(self) -> np.ndarray:
        assert self.game is not None and self.encoder is not None
        return self.encoder.encode(self.game, self.agent_seat)

    def _info(self) -> Dict[str, Any]:
        game = self.game
        assert game is not None
        info: Dict[str, Any] = {
            "action_mask": game.action_mask(),
            "round": game.state.round_index,
            "current_player": game.current_player,
        }
        if game.pending is not None:
            info["decision"] = game.pending.kind
            info["prompt"] = game.pending.prompt
            info["legal_actions"] = game.pending.action_ids()
            info["option_labels"] = {o.action_id: o.label for o in game.pending.options}
        return info
