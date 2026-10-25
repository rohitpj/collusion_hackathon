"""Descending-clock procurement auction. Pure functions, no I/O.

Each firm submits a reservation price r_i: it stays in while the clock price p >= r_i and
drops out once p < r_i. The last firm standing wins and is paid the clock price at which the
second-to-last firm dropped (about the second-lowest reservation). Honest r_i = cost_i.
Standing down = r_i = cap (drop out at the first tick). If every rival stands down the
winner is paid the cap.
"""
import math
import random
from dataclasses import dataclass


@dataclass
class Outcome:
    winner: int
    price: int


def clamp(r, cap):
    return max(0, min(int(r), cap))


def run_auction(reservations, cap, step, rng):
    """reservations: list[int] indexed by firm. Returns Outcome."""
    r = [clamp(x, cap) for x in reservations]
    order = sorted(range(len(r)), key=lambda i: (r[i], rng.random()))
    winner, runner_up = order[0], order[1]
    price = min(cap, math.ceil(r[runner_up] / step) * step)
    return Outcome(winner, price)


def competitive_price(costs, cap, step):
    """Price if everyone bids honestly: second-lowest cost on the clock ladder."""
    c = sorted(costs)
    return min(cap, math.ceil(c[1] / step) * step)


def theory_ring_price(costs, ring, cap, step):
    """Price when the ring is fully obedient. The ring acts as one bidder whose cost is its cheapest
    member's; the price is the second-lowest of {ring minimum, each outsider}. No outsiders -> cap."""
    if len(ring) < 2:
        return competitive_price(costs, cap, step)
    outsiders = [c for i, c in enumerate(costs) if i not in ring]
    if not outsiders:
        return cap
    c = sorted([min(costs[i] for i in ring)] + outsiders)
    return min(cap, math.ceil(c[1] / step) * step)


def draw_costs(n, lo, hi, rng, step):
    return [int(round(rng.uniform(lo, hi) / step)) * step for _ in range(n)]
