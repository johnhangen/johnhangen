"""Rules-level tests: mat economics, power effects, goals and scoring."""

import random

from conftest import by_kind, drive, make_game

from wingspan_rl.bonus import BONUS_CARDS
from wingspan_rl.constants import (
    EGG_COST_BY_COLUMN,
    HABITAT_ACTION_VALUE,
    Food,
    Habitat,
    NestType,
    Timing,
)
from wingspan_rl.goals import GOALS, score_goal
from wingspan_rl.powers import Power
from wingspan_rl.scoring import score_player
from wingspan_rl.state import DIE_FACES, Birdfeeder


def place(game, player, habitat, card_id, eggs=0):
    board = game.state.board(player)
    placed = board.place(game.cards[card_id], habitat)
    placed.eggs = eggs
    return placed


# --- player mat -----------------------------------------------------------


def test_setup_gives_every_player_five_tokens_and_one_bonus_card():
    game = make_game(num_players=3, seed=9)
    rng = random.Random(0)
    while game.pending is not None and game.pending.kind != "main":
        game.step(rng.choice(game.legal_action_ids()))
    for board in game.state.players:
        assert len(board.hand) + board.total_food() == 5
        assert len(board.bonus_cards) == 1


def test_action_value_grows_with_the_row():
    game = make_game(seed=1)
    board = game.state.board(0)
    for count in range(5):
        assert board.action_value(Habitat.FOREST) == HABITAT_ACTION_VALUE[Habitat.FOREST][count]
        place(game, 0, Habitat.FOREST, count)
    # A full row keeps the strongest action.
    assert board.action_value(Habitat.FOREST) == 3


def test_exchanges_unlock_as_the_row_fills():
    game = make_game(seed=1)
    board = game.state.board(0)
    assert board.exchanges_available(Habitat.WETLAND) == 0
    place(game, 0, Habitat.WETLAND, 0)
    assert board.exchanges_available(Habitat.WETLAND) == 1
    for card_id in (1, 2, 3):
        place(game, 0, Habitat.WETLAND, card_id)
    assert board.exchanges_available(Habitat.WETLAND) == 2


def test_playing_into_a_later_column_costs_eggs():
    game = make_game(seed=1)
    board = game.state.board(0)
    forest = [c for c in game.cards if c.habitats == (Habitat.FOREST,)][:2]
    filler = place(game, 0, Habitat.FOREST, forest[0].id, eggs=2)
    board.hand = [forest[1].id]
    board.food = [5, 5, 5, 5, 5]
    drive(game._play_bird_flow(0, card_id=forest[1].id),
          [by_kind("habitat"), by_kind("bird"), by_kind("payment")])
    assert board.row_size(Habitat.FOREST) == 2
    assert EGG_COST_BY_COLUMN[1] == 1
    assert filler.eggs == 1  # one egg paid for the second column
    assert board.hand == []


def test_a_bird_cannot_be_played_without_enough_eggs():
    game = make_game(seed=1)
    board = game.state.board(0)
    forest = [c for c in game.cards if Habitat.FOREST in c.habitats][:2]
    place(game, 0, Habitat.FOREST, forest[0].id, eggs=0)
    board.hand = [forest[1].id]
    board.food = [5, 5, 5, 5, 5]
    assert game._playable_habitats(board, forest[1]) == [
        h for h in forest[1].habitats if h is not Habitat.FOREST
    ]


# --- birdfeeder -----------------------------------------------------------


def test_feeder_never_shows_a_single_face():
    for seed in range(20):
        feeder = Birdfeeder(random.Random(seed))
        assert len(set(feeder.dice)) > 1


def test_taking_the_last_die_refills_the_feeder():
    feeder = Birdfeeder(random.Random(0))
    while len(feeder.dice) > 1:
        feeder.take(feeder.dice[0])
    feeder.take(feeder.dice[0])
    assert len(feeder.dice) == 5


def test_combined_die_face_grants_both_foods():
    game = make_game(seed=1)
    board = game.state.board(0)
    combo = len(DIE_FACES) - 1
    game.state.feeder.dice = [combo, 0]
    drive(game._take_die(0, "test"),
          [lambda d: next(o for o in d.options if o.payload == combo)])
    assert board.food[int(Food.INVERTEBRATE)] == 1
    assert board.food[int(Food.SEED)] == 1


# --- powers ---------------------------------------------------------------


def test_cache_food_power_stores_food_on_the_card():
    game = make_game(seed=2)
    placed = place(game, 0, Habitat.FOREST, 0)
    power = Power("cache_food", Timing.BROWN, {"food": "seed", "count": 2})
    drive(game._apply_power(0, placed, power))
    assert placed.cached[int(Food.SEED)] == 2
    assert game.state.board(0).food[int(Food.SEED)] == 0


def test_lay_eggs_power_respects_capacity():
    game = make_game(seed=2)
    card = next(c for c in game.cards if c.egg_capacity == 2)
    placed = place(game, 0, Habitat.GRASSLAND, card.id)
    power = Power("lay_eggs", Timing.BROWN, {"count": 5, "target": "this"})
    drive(game._apply_power(0, placed, power))
    assert placed.eggs == card.egg_capacity


def test_tuck_removes_the_card_from_the_game_and_pays_an_egg():
    game = make_game(seed=2)
    board = game.state.board(0)
    card = next(c for c in game.cards if c.egg_capacity >= 2)
    placed = place(game, 0, Habitat.FOREST, card.id)
    board.hand = [10, 11]
    discard_before = len(game.state.discard)
    power = Power("tuck_from_hand", Timing.BROWN, {"count": 1, "then": "egg"})
    drive(game._apply_power(0, placed, power), [0])
    assert placed.tucked == 1
    assert placed.eggs == 1
    assert len(board.hand) == 1
    assert len(game.state.discard) == discard_before


def test_predator_hunt_succeeds_on_a_small_bird():
    game = make_game(seed=2)
    placed = place(game, 0, Habitat.FOREST, 0)
    small = min(game.cards, key=lambda c: c.wingspan)
    game.state.deck = [small.id]
    power = Power("predator_hunt", Timing.BROWN,
                  {"threshold": small.wingspan + 1, "reward": "cache", "food": "fish"})
    drive(game._apply_power(0, placed, power))
    assert placed.cached[int(Food.FISH)] == 1


def test_predator_hunt_fails_on_a_large_bird():
    game = make_game(seed=2)
    placed = place(game, 0, Habitat.FOREST, 0)
    large = max(game.cards, key=lambda c: c.wingspan)
    game.state.deck = [large.id]
    power = Power("predator_hunt", Timing.BROWN,
                  {"threshold": 20, "reward": "tuck", "food": "rodent"})
    drive(game._apply_power(0, placed, power))
    assert placed.tucked == 0
    assert game.state.discard[-1] == large.id


def test_repeat_brown_runs_another_birds_power():
    game = make_game(seed=2)
    donor_card = next(c for c in game.cards if Habitat.FOREST in c.habitats)
    donor = place(game, 0, Habitat.FOREST, donor_card.id)
    repeater = place(game, 0, Habitat.FOREST, next(
        c.id for c in game.cards if Habitat.FOREST in c.habitats and c.id != donor_card.id))
    game.cards[donor.card_id].power = Power(
        "gain_food_supply", Timing.BROWN, {"food": "fish", "count": 1}
    )
    drive(game._apply_power(0, repeater, Power("repeat_brown", Timing.BROWN)), [0])
    assert game.state.board(0).food[int(Food.FISH)] == 1


def test_all_players_gain_food():
    game = make_game(num_players=3, seed=2)
    placed = place(game, 0, Habitat.WETLAND, 0)
    before = [b.food[int(Food.SEED)] for b in game.state.players]
    power = Power("all_players_gain_food", Timing.BROWN, {"food": "seed"})
    drive(game._apply_power(0, placed, power))
    after = [b.food[int(Food.SEED)] for b in game.state.players]
    assert after == [b + 1 for b in before]


def test_gain_bonus_card_keeps_exactly_one():
    game = make_game(seed=2)
    board = game.state.board(0)
    board.bonus_cards = []
    placed = place(game, 0, Habitat.FOREST, 0)
    drive(game._apply_power(0, placed, Power("gain_bonus_card", Timing.WHITE)), [0])
    assert len(board.bonus_cards) == 1
    assert len(game.state.bonus_discard) == 1


def test_play_extra_bird_power_plays_from_hand():
    game = make_game(seed=2)
    board = game.state.board(0)
    card = next(c for c in game.cards if Habitat.WETLAND in c.habitats)
    board.hand = [card.id]
    board.food = [5, 5, 5, 5, 5]
    placed = place(game, 0, Habitat.FOREST, next(
        c.id for c in game.cards if Habitat.FOREST in c.habitats and c.id != card.id))
    power = Power("play_extra_bird", Timing.WHITE, {"habitat": "wetland"})
    drive(game._apply_power(0, placed, power), [0, 0, 0])
    assert board.row_size(Habitat.WETLAND) == 1
    assert board.hand == []


def test_pink_power_triggers_on_an_opponent_action_only_once():
    game = make_game(num_players=2, seed=2)
    card = next(c for c in game.cards if c.egg_capacity >= 3)
    placed = place(game, 1, Habitat.GRASSLAND, card.id)
    game.cards[card.id].power = Power(
        "on_opponent_action", Timing.PINK, {"trigger": "gain_food", "effect": "lay_egg"}
    )
    drive(game._trigger_pink(0, "gain_food"))
    assert placed.eggs == 1
    drive(game._trigger_pink(0, "gain_food"))
    assert placed.eggs == 1  # already used between turns
    placed.pink_used = False
    drive(game._trigger_pink(0, "draw_cards"))
    assert placed.eggs == 1  # wrong trigger


# --- goals and scoring -----------------------------------------------------


def test_goal_places_and_ties():
    assert score_goal([3, 1, 0], 0) == [4, 1, 0]
    assert score_goal([2, 2, 1], 0) == [4, 4, 0]      # tie consumes two places
    assert score_goal([0, 0, 0], 2) == [0, 0, 0]      # zero never scores
    assert score_goal([5, 4, 3, 2], 3) == [7, 4, 3, 0]


def test_goal_counters_read_the_board():
    game = make_game(seed=3)
    board = game.state.board(0)
    card = next(c for c in game.cards if c.nest is NestType.BOWL)
    place(game, 0, Habitat.FOREST, card.id, eggs=2)
    forest_birds = next(g for g in GOALS if g.name == "birds in forest")
    bowl_eggs = next(g for g in GOALS if g.name == "eggs in bowl nests")
    assert forest_birds.count(board, game.cards) == 1
    assert bowl_eggs.count(board, game.cards) == 2


def test_bonus_card_tiers():
    game = make_game(seed=3)
    board = game.state.board(0)
    bonus = next(b for b in BONUS_CARDS if b.name == "Forest Ecologist")
    forest = [c for c in game.cards if Habitat.FOREST in c.habitats]
    for card in forest[:2]:
        place(game, 0, Habitat.FOREST, card.id)
    assert bonus.score(board, game.cards) == 0
    place(game, 0, Habitat.FOREST, forest[2].id)
    assert bonus.score(board, game.cards) == 3
    for card in forest[3:5]:
        place(game, 0, Habitat.FOREST, card.id)
    assert bonus.score(board, game.cards) == 6


def test_final_score_adds_every_component():
    game = make_game(seed=3)
    board = game.state.board(0)
    board.bonus_cards = []
    card = next(c for c in game.cards
                if c.points >= 3 and c.egg_capacity >= 2 and c.power.timing is not Timing.GAME_END)
    placed = place(game, 0, Habitat.FOREST, card.id, eggs=2)
    placed.cached = (0, 3, 0, 0, 0)
    placed.tucked = 1
    board.round_goal_points = [4, 0, 2, 0]
    score = score_player(board, game.cards, game.bonus_cards)
    assert score.birds == card.points
    assert score.eggs == 2
    assert score.cached_food == 3
    assert score.tucked_cards == 1
    assert score.goals == 6
    assert score.total == card.points + 2 + 3 + 1 + 6


def test_game_end_power_scores_points():
    game = make_game(seed=3)
    board = game.state.board(0)
    board.bonus_cards = []
    card = game.cards[0]
    card.power = Power("end_points_per", Timing.GAME_END,
                       {"per": "eggs_on_this", "amount": 2})
    place(game, 0, Habitat.FOREST, card.id, eggs=3)
    score = score_player(board, game.cards, game.bonus_cards)
    assert score.end_powers == 6
