import random

import numpy as np

from conftest import make_game, play_random

from wingspan_rl.constants import Habitat, MAX_OPTIONS
from wingspan_rl.observation import (
    CARD_FEATURES,
    OBS_SIZE,
    ObservationEncoder,
    section_slices,
)


def test_observation_shape_and_range():
    game = make_game(seed=1)
    encoder = ObservationEncoder(game.cards)
    obs = encoder.encode(game, 0)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert obs.min() >= 0.0


def test_sections_tile_the_vector_exactly():
    slices = section_slices()
    assert list(slices)[0] == "global"
    end = 0
    for name, sl in slices.items():
        assert sl.start == end, name
        end = sl.stop
    assert end == OBS_SIZE


def test_observation_stays_finite_through_a_whole_game():
    game = make_game(num_players=3, seed=4)
    encoder = ObservationEncoder(game.cards)
    rng = random.Random(4)
    while not game.done:
        for player in range(3):
            obs = encoder.encode(game, player)
            assert np.isfinite(obs).all()
        game.step(rng.choice(game.legal_action_ids()))


def test_board_section_reflects_played_birds():
    game = make_game(seed=1)
    encoder = ObservationEncoder(game.cards)
    board_slice = section_slices()["board"]
    before = encoder.encode(game, 0)[board_slice]
    assert not before.any()
    card = next(c for c in game.cards if Habitat.FOREST in c.habitats)
    placed = game.state.board(0).place(card, Habitat.FOREST)
    placed.eggs = 2
    after = encoder.encode(game, 0)[board_slice]
    assert after.any()
    # the occupancy flag sits right after the card features of the first slot
    assert after[CARD_FEATURES] == 1.0


def test_option_section_encodes_the_pending_choices():
    game = make_game(seed=1)
    encoder = ObservationEncoder(game.cards)
    options_slice = section_slices()["options"]
    obs = encoder.encode(game, game.pending.player)
    per = (options_slice.stop - options_slice.start) // MAX_OPTIONS
    active = [i for i in range(MAX_OPTIONS)
              if obs[options_slice.start + i * per: options_slice.start + (i + 1) * per].any()]
    assert len(active) == len(game.pending.options)


def test_encoder_is_perspective_dependent():
    game = make_game(num_players=2, seed=6)
    encoder = ObservationEncoder(game.cards)
    play_random(game, random.Random(6))
    assert not np.array_equal(encoder.encode(game, 0), encoder.encode(game, 1))


def test_distinct_payment_options_encode_differently():
    """An agent must be able to tell 'pay 2 seeds' from 'pay 1 seed 1 fruit'."""
    import numpy as np

    from wingspan_rl.cards import Cost
    from wingspan_rl.constants import MAX_OPTIONS
    from wingspan_rl.decisions import OPTION_KINDS
    from wingspan_rl.observation import OPTION_CARD_FEATURES

    game = make_game(seed=1)
    board = game.state.board(0)
    card = next(c for c in game.cards if Habitat.FOREST in c.habitats)
    card.costs = (Cost(wild=2),)
    board.hand = [card.id]
    board.food = [0, 2, 0, 2, 0]

    flow = game._play_bird_flow(0, card_id=card.id)
    decision = flow.send(None)              # habitat
    decision = flow.send(decision.options[0].payload)
    while decision.kind != "play_pay":      # skip any egg payment
        decision = flow.send(decision.options[0].payload)
    payments = [o for o in decision.options if o.kind == "payment"]
    assert len(payments) >= 2

    game.pending = decision
    encoder = ObservationEncoder(game.cards)
    obs = encoder.encode(game, 0)
    options_slice = section_slices()["options"]
    per = (options_slice.stop - options_slice.start) // MAX_OPTIONS
    blocks = [
        obs[options_slice.start + i * per: options_slice.start + (i + 1) * per]
        for i in range(len(payments))
    ]
    assert not np.array_equal(blocks[0], blocks[1])
    assert len(OPTION_KINDS) + 2 + OPTION_CARD_FEATURES == per
