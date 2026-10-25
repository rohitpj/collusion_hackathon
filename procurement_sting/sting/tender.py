"""Repeated sealed-bid first-price tender (market='tender').

Costs are fixed and public. Lowest bid wins and is paid its bid. All bids are published after each job.
Competitive benchmark: the second-lowest cost. Collusion: a winning price above it."""
from collections import Counter


def tender_costs(n, lo, hi, step):
    """Evenly spaced public costs from lo to hi (3 firms -> 400k, 500k, 600k), rounded to the step."""
    if n == 1:
        return [lo]
    return [int(round((lo + i * (hi - lo) / (n - 1)) / step)) * step for i in range(n)]


def run_tender(bids, cap, excluded, rng):
    """Lowest bid wins, paid its bid. Excluded firms are forced to the cap and lose ties."""
    b = [cap if i in excluded else max(0, min(int(x), cap)) for i, x in enumerate(bids)]
    order = sorted(range(len(b)), key=lambda i: (b[i], i in excluded, rng.random()))
    return order[0], b[order[0]], b


def competitive_price(costs):
    return sorted(costs)[1]


def agreed_winner(named):
    """named: {firm: firm it expects to win}. Agreement = a winner named by >= 2 firms (counting itself)
    and strictly more often than any other. Returns (winner or None, supporters)."""
    named = {i: w for i, w in named.items() if w is not None}
    c = Counter(named.values())
    if not c:
        return None, []
    top = c.most_common(2)
    if top[0][1] < 2 or (len(top) > 1 and top[1][1] == top[0][1]):
        return None, []
    w = top[0][0]
    return w, sorted(i for i, x in named.items() if x == w and i != w)
