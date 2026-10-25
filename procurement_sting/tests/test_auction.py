import random
from sting.auction import run_auction, competitive_price, theory_ring_price

CAP, STEP = 1_000_000, 10_000
COSTS = [400_000, 500_000, 600_000, 700_000, 800_000]


def test_competitive():
    o = run_auction(COSTS, CAP, STEP, random.Random(0))
    assert (o.winner, o.price) == (0, 500_000)


def test_ring_table():
    # the spec's table: ring of cheapest k -> price 500k,600k,800k,1m (k=1 is "nobody")
    for k, want in [(2, 600_000), (4, 800_000), (5, 1_000_000)]:
        ring = set(range(k))
        res = [COSTS[0] if i == 0 else CAP for i in range(5)]
        res = [r if i in ring else COSTS[i] for i, r in enumerate(res)]
        o = run_auction(res, CAP, STEP, random.Random(0))
        assert o.winner == 0 and o.price == want == theory_ring_price(COSTS, ring, CAP, STEP)


def test_random_ring_theory_matches_play():
    rng = random.Random(1)
    for _ in range(500):
        costs = [rng.randrange(40, 80) * 10_000 for _ in range(5)]
        ring = set(rng.sample(range(5), rng.randint(2, 5)))
        w = min(ring, key=lambda i: costs[i])
        res = [costs[i] if (i not in ring or i == w) else CAP for i in range(5)]
        o = run_auction(res, CAP, STEP, random.Random(0))
        assert o.price == theory_ring_price(costs, ring, CAP, STEP)
