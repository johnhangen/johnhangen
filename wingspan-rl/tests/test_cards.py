from wingspan_rl.cards import BirdCard, Cost, enumerate_payments, food_counts, load_deck
from wingspan_rl.constants import Food, Habitat, NestType
from wingspan_rl.powers import Power, POWER_KINDS


def test_bundled_deck_loads_and_validates():
    deck = load_deck()
    assert len(deck) == 170
    assert [c.id for c in deck] == list(range(len(deck)))
    assert all(c.habitats for c in deck)
    assert all(1 <= c.egg_capacity <= 6 for c in deck)
    assert all(c.power.kind in POWER_KINDS for c in deck)


def test_deck_covers_every_habitat_and_nest():
    deck = load_deck()
    for habitat in Habitat:
        assert sum(1 for c in deck if habitat in c.habitats) >= 20
    for nest in NestType:
        assert any(c.nest is nest for c in deck)


def test_card_roundtrip():
    card = load_deck()[7]
    clone = BirdCard.from_dict(card.to_dict())
    assert clone.to_dict() == card.to_dict()


def test_exact_cost_payment():
    cost = Cost(food=food_counts({Food.SEED: 2}))
    assert enumerate_payments((0, 2, 0, 0, 0), [cost]) == [(0, 2, 0, 0, 0)]
    assert enumerate_payments((0, 1, 0, 0, 0), [cost]) == []


def test_wild_cost_enumerates_distinct_payments():
    cost = Cost(food=food_counts({Food.SEED: 1}), wild=1)
    payments = enumerate_payments((1, 2, 0, 0, 0), [cost])
    assert set(payments) == {(0, 2, 0, 0, 0), (1, 1, 0, 0, 0)}


def test_alternative_costs_are_merged_and_deduplicated():
    costs = (Cost(food=food_counts({Food.SEED: 1})),
             Cost(food=food_counts({Food.FRUIT: 1})))
    payments = enumerate_payments((0, 1, 0, 1, 0), costs)
    assert payments == [(0, 0, 0, 1, 0), (0, 1, 0, 0, 0)]


def test_food_types_and_min_cost():
    card = BirdCard(
        id=0, name="Test", points=3, habitats=(Habitat.FOREST,),
        costs=(Cost(food=food_counts({Food.FISH: 2})),
               Cost(food=food_counts({Food.RODENT: 1}))),
        nest=NestType.STAR, egg_capacity=3, wingspan=50,
        power=Power("draw_cards", __import__("wingspan_rl").Timing.BROWN, {"count": 1}),
    )
    assert card.food_types == {Food.FISH, Food.RODENT}
    assert card.min_cost == 1
    assert card.nest_matches(NestType.BOWL)  # star nest is wild
