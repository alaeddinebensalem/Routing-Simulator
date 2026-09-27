"""Route optimisation.

Pipeline:

1. `RoutePlan.build` runs A* (Manhattan heuristic) for every pair the robot
   could ever travel: start cell -> each pickup, each pickup -> its own
   dropoff, and every dropoff -> every other pickup.
2. `random_tour_moves` serves every reachable customer in a random order; its
   length is the natural cap for the robot's move budget.
3. `solve` is a backtracking search over customer orderings, sped up five ways:
   - a seeded initial incumbent (greedy + local search) so real pruning starts
     from node zero instead of from -1
   - lookahead child ordering (best value-per-actual-travel-from-here first)
   - a cheap bound, a fractional-knapsack bound, and a matching-based bound,
     tried in that order and only as far as needed to prune
   - an O(log n) dead-end pre-check before descending into a branch
   - dominance memoisation on (last customer, visited set): a state reached
     with no less travel than a previous visit is dropped without recursing
4. `brute_force` enumerates the same space with none of the above, so the
   benchmark panel can show what the speedups are worth.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum

from grid import Cell, Customer, Grid
from pathfinding import astar, path_moves

INF = float("inf")


class Objective(Enum):
    CUSTOMERS = "Customers served"
    LOADED_MOVES = "Loaded moves"


@dataclass
class RoutePlan:
    """All pairwise shortest paths the router needs."""

    depot: Cell
    customers: list[Customer]
    depot_path: list[list[Cell] | None]
    loaded_path: list[list[Cell] | None]
    gap_path: list[list[list[Cell] | None]]
    depot_dist: list[float]
    loaded_dist: list[float]
    gap_dist: list[list[float]]
    # Cheapest way the robot could ever arrive at customer i: either straight
    # from its start cell, or out of the closest other customer's dropoff.
    min_arrival: list[float]

    @property
    def size(self) -> int:
        return len(self.customers)

    def reachable(self, i: int) -> bool:
        return self.depot_dist[i] < INF and self.loaded_dist[i] < INF

    @classmethod
    def build(cls, grid: Grid, depot: Cell) -> "RoutePlan":
        customers = list(grid.customers)
        k = len(customers)

        depot_path = [astar(grid, depot, c.start) for c in customers]
        loaded_path = [astar(grid, c.start, c.end) for c in customers]
        gap_path: list[list[list[Cell] | None]] = [[None] * k for _ in range(k)]
        for i, ci in enumerate(customers):
            for j, cj in enumerate(customers):
                if i != j:
                    gap_path[i][j] = astar(grid, ci.end, cj.start)

        depot_dist = [path_moves(p) for p in depot_path]
        loaded_dist = [path_moves(p) for p in loaded_path]
        gap_dist = [[path_moves(gap_path[i][j]) for j in range(k)] for i in range(k)]

        # Per-customer arrival lower bound, precomputed once so the search pays
        # nothing for it.  Attributing the hop to the customer being *entered*
        # (rather than the one being left) keeps the accounting exact: every
        # served customer, including the first, has exactly one arrival, so
        # weight[i] never overcharges any leg that could actually reach it.
        min_arrival: list[float] = []
        for i in range(k):
            best = depot_dist[i]
            for j in range(k):
                if j != i and gap_dist[j][i] < best:
                    best = gap_dist[j][i]
            min_arrival.append(0.0 if best is INF else best)

        return cls(
            depot=depot,
            customers=customers,
            depot_path=depot_path,
            loaded_path=loaded_path,
            gap_path=gap_path,
            depot_dist=depot_dist,
            loaded_dist=loaded_dist,
            gap_dist=gap_dist,
            min_arrival=min_arrival,
        )

    # ------------------------------------------------------------------
    def leg_cost(self, prev: int, i: int) -> float:
        """Moves to reach customer i's pickup from `prev` (-1 = start) and deliver."""
        approach = self.depot_dist[i] if prev < 0 else self.gap_dist[prev][i]
        return approach + self.loaded_dist[i]

    def tour_moves(self, order: list[int]) -> float:
        total = 0.0
        prev = -1
        for i in order:
            total += self.leg_cost(prev, i)
            prev = i
        return total


def random_tour_moves(plan: RoutePlan, rng: random.Random | None = None) -> float:
    """Moves used by one randomly ordered tour that serves every customer."""
    rng = rng or random
    order = [i for i in range(plan.size) if plan.reachable(i)]
    rng.shuffle(order)
    return plan.tour_moves(order)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
@dataclass
class Leg:
    kind: str  # "approach" | "delivery"
    customer: int | None  # index into plan.customers
    path: list[Cell]

    @property
    def moves(self) -> int:
        return max(0, len(self.path) - 1)


@dataclass
class Solution:
    order: list[int] = field(default_factory=list)
    value: float = 0.0
    moves: float = 0.0
    legs: list[Leg] = field(default_factory=list)
    optimal: bool = True
    nodes: int = 0
    elapsed: float = 0.0
    budget: float = 0.0
    objective: Objective = Objective.CUSTOMERS
    label: str = "branch and bound"


class _Stop(Exception):
    """Raised to unwind the search when a node or time limit is hit."""


def customer_values(plan: RoutePlan, objective: Objective) -> list[float]:
    if objective is Objective.CUSTOMERS:
        return [1.0 if plan.reachable(i) else 0.0 for i in range(plan.size)]
    return [plan.loaded_dist[i] if plan.reachable(i) else 0.0 for i in range(plan.size)]


def customer_weights(plan: RoutePlan) -> list[float]:
    """Optimistic travel each customer must consume: arrival hop + loaded run."""
    return [
        max(plan.loaded_dist[i] + plan.min_arrival[i], 1e-9) if plan.reachable(i) else INF
        for i in range(plan.size)
    ]


# ----------------------------------------------------------------------
# Fenwick-backed views over the customers still available
# ----------------------------------------------------------------------
class AvailableSet:
    """Customers not yet used, held in descending value-per-travel order.

    Three Fenwick trees (count, weight, value) over that fixed order give
    O(log n) instead of an O(n) scan:

    * `best_rate` is the first still-available slot (tier 1 bound).
    * `knapsack_bound` walks the trees to the longest prefix that fits the
      capacity, reads off its value, and adds one fractional item (tier 2).

    `remove`/`restore` are the incremental update and rollback the search does
    when it enters and leaves a branch.
    """

    def __init__(self, weights: list[float], values: list[float]) -> None:
        self.n = len(weights)
        self.weights = weights
        self.values = values
        size = self.n + 1
        self._count = [0] * size
        self._weight = [0.0] * size
        self._value = [0.0] * size
        for i in range(self.n):
            self._update(i, 1, weights[i], values[i])
        self._log = 1
        while self._log * 2 <= self.n:
            self._log *= 2

    def _update(self, slot: int, d_count: int, d_weight: float, d_value: float) -> None:
        i = slot + 1
        while i <= self.n:
            self._count[i] += d_count
            self._weight[i] += d_weight
            self._value[i] += d_value
            i += i & -i

    def remove(self, slot: int) -> None:
        self._update(slot, -1, -self.weights[slot], -self.values[slot])

    def restore(self, slot: int) -> None:
        self._update(slot, 1, self.weights[slot], self.values[slot])

    def kth(self, k: int) -> int:
        """Slot of the k-th still-available customer (1-based), or -1."""
        pos, step, rest = 0, self._log, k
        while step:
            nxt = pos + step
            if nxt <= self.n and self._count[nxt] < rest:
                rest -= self._count[nxt]
                pos = nxt
            step //= 2
        return pos if pos < self.n else -1

    def best_rate(self) -> float:
        slot = self.kth(1)
        if slot < 0:
            return 0.0
        return self.values[slot] / self.weights[slot]

    def knapsack_bound(self, capacity: float) -> float:
        """Fractional knapsack over the remaining customers, in O(log n).

        Removed customers sit at zero weight and zero value, so the tree walk
        steps over them for free.
        """
        if capacity <= 0 or self.n == 0:
            return 0.0
        pos, step = 0, self._log
        rest, gained, taken = capacity, 0.0, 0
        while step:
            nxt = pos + step
            if nxt <= self.n and self._weight[nxt] <= rest + 1e-9:
                rest -= self._weight[nxt]
                gained += self._value[nxt]
                taken += self._count[nxt]
                pos = nxt
            step //= 2
        slot = self.kth(taken + 1)
        if slot >= 0 and rest > 0:
            gained += self.values[slot] * rest / self.weights[slot]
        return gained


class CheapestTracker:
    """Cheapest still-available customer's weight, in O(log n).

    Used for the dead-end pre-check: right after tentatively taking a
    customer, is there anything left that could possibly fit in the budget
    that remains?  If not, there is no point recursing a level deeper just to
    have the child discover that on its own.
    """

    def __init__(self, weights_ascending: list[float]) -> None:
        self.n = len(weights_ascending)
        self.weights = weights_ascending
        self._count = [0] * (self.n + 1)
        for i in range(self.n):
            self._update(i, 1)
        self._log = 1
        while self._log * 2 <= self.n:
            self._log *= 2

    def _update(self, slot: int, delta: int) -> None:
        i = slot + 1
        while i <= self.n:
            self._count[i] += delta
            i += i & -i

    def remove(self, slot: int) -> None:
        self._update(slot, -1)

    def restore(self, slot: int) -> None:
        self._update(slot, 1)

    def _kth(self, k: int) -> int:
        pos, step, rest = 0, self._log, k
        while step:
            nxt = pos + step
            if nxt <= self.n and self._count[nxt] < rest:
                rest -= self._count[nxt]
                pos = nxt
            step //= 2
        return pos if pos < self.n else -1

    def min_weight(self) -> float:
        slot = self._kth(1)
        return self.weights[slot] if slot >= 0 else INF


def _scan_knapsack_bound(
    by_rate: list[int], used, weights, values, capacity: float
) -> float:
    """Reference O(n) implementation, kept for the tests to check against."""
    gained = 0.0
    for i in by_rate:
        is_used = used[i] if not isinstance(used, int) else bool((used >> i) & 1)
        if is_used or capacity <= 0:
            continue
        if weights[i] <= capacity:
            gained += values[i]
            capacity -= weights[i]
        else:
            gained += values[i] * capacity / weights[i]
            capacity = 0.0
    return gained


# ----------------------------------------------------------------------
# Tier 3: matching-based bound
# ----------------------------------------------------------------------
def hungarian_min_cost(cost: list[list[float]]) -> tuple[float, list[int]]:
    """Minimum-cost assignment of each row to a distinct column (n <= m).

    Returns (total_cost, assignment) where assignment[i] is the column index
    matched to row i.  O(n^2 * m) Kuhn-Munkres with potentials, generalised to
    rectangular matrices.
    """
    n = len(cost)
    if n == 0:
        return 0.0, []
    m = len(cost[0])
    BIG = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [BIG] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = BIG
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [0] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    total = sum(cost[i][assignment[i]] for i in range(n))
    return total, assignment


TIER3_MAX_REMAINING = 26  # Hungarian is O(m^2 * (m+1)); skip it above this size.
_BIG_COST = 1e6


def tier3_bound(
    plan: RoutePlan,
    mask: int,
    prev: int,
    values: list[float],
    live: list[int],
    remaining_budget: float,
) -> float:
    """Matching-based relaxation, tried only after the cheaper bounds fail.

    Every remaining customer needs exactly one arrival, and that arrival must
    come from a distinct provider: the current position (usable once) or
    another remaining customer's dropoff (also usable once, unlike the
    per-customer independent-minimum used by the cheap bound, which lets
    several customers claim the same dropoff at no cost). The minimum-cost
    assignment of customers to providers is therefore a valid lower bound on
    the travel needed to visit everyone still left, in ANY order — a standard
    assignment-problem relaxation of the underlying routing problem.
    """
    rem = [i for i in live if not (mask >> i) & 1]
    m = len(rem)
    if m == 0:
        return 0.0
    if m > TIER3_MAX_REMAINING:
        return INF  # too expensive here; the cheaper bounds carry this node

    cols = [-1] + rem
    cost: list[list[float]] = []
    for i in rem:
        row = []
        for c in cols:
            if c == -1:
                arrival = plan.depot_dist[i] if prev < 0 else plan.gap_dist[prev][i]
            elif c == i:
                arrival = INF
            else:
                arrival = plan.gap_dist[c][i]
            row.append(_BIG_COST if arrival == INF else arrival)
        cost.append(row)

    _, assignment = hungarian_min_cost(cost)
    weight2 = [plan.loaded_dist[rem[idx]] + cost[idx][assignment[idx]] for idx in range(m)]
    order2 = sorted(range(m), key=lambda idx: -(values[rem[idx]] / max(weight2[idx], 1e-9)))

    cap = remaining_budget
    gained = 0.0
    for idx in order2:
        if cap <= 0:
            break
        w, v = weight2[idx], values[rem[idx]]
        if w <= cap:
            gained += v
            cap -= w
        else:
            gained += v * cap / max(w, 1e-9)
            cap = 0.0
    return gained


# ----------------------------------------------------------------------
# Initial incumbent: greedy + local search, seeded before the tree opens
# ----------------------------------------------------------------------
def _greedy_order(
    plan: RoutePlan, by_rate: list[int], budget: float
) -> list[int]:
    order: list[int] = []
    travel = 0.0
    for i in by_rate:
        step = plan.leg_cost(order[-1] if order else -1, i)
        if step < INF and travel + step <= budget:
            order.append(i)
            travel += step
    return order


def _two_opt(plan: RoutePlan, order: list[int]) -> list[int]:
    improved = True
    while improved:
        improved = False
        n = len(order)
        best_cost = plan.tour_moves(order)
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                cost = plan.tour_moves(candidate)
                if cost < best_cost - 1e-9:
                    order, best_cost, improved = candidate, cost, True
    return order


def _or_opt(plan: RoutePlan, order: list[int]) -> list[int]:
    improved = True
    while improved:
        improved = False
        best_cost = plan.tour_moves(order)
        for i in range(len(order)):
            cust = order[i]
            rest = order[:i] + order[i + 1 :]
            for j in range(len(rest) + 1):
                candidate = rest[:j] + [cust] + rest[j:]
                cost = plan.tour_moves(candidate)
                if cost < best_cost - 1e-9:
                    order, best_cost, improved = candidate, cost, True
                    break
            if improved:
                break
    return order


def _insert_extras(plan: RoutePlan, order: list[int], live: list[int], budget: float) -> list[int]:
    present = set(order)
    changed = True
    while changed:
        changed = False
        for i in live:
            if i in present:
                continue
            best_cost, best_order = None, None
            for j in range(len(order) + 1):
                candidate = order[:j] + [i] + order[j:]
                cost = plan.tour_moves(candidate)
                if cost <= budget and (best_cost is None or cost < best_cost):
                    best_cost, best_order = cost, candidate
            if best_order is not None:
                order, changed = best_order, True
                present.add(i)
    return order


def build_initial_incumbent(
    plan: RoutePlan,
    values: list[float],
    by_rate: list[int],
    live: list[int],
    budget: float,
) -> tuple[list[int], float, float]:
    """Greedy-by-rate, then 2-opt / or-opt / insertion, so real pruning starts
    from a strong bound instead of from nothing."""
    order = _greedy_order(plan, by_rate, budget)
    order = _two_opt(plan, order)
    order = _or_opt(plan, order)
    order = _insert_extras(plan, order, live, budget)
    order = _two_opt(plan, order)
    value = sum(values[i] for i in order)
    moves = plan.tour_moves(order)
    return order, value, moves


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
def solve(
    plan: RoutePlan,
    objective: Objective,
    budget: float,
    node_limit: int = 2_000_000,
    time_limit: float | None = None,
    progress: dict | None = None,
) -> Solution:
    """Maximise `objective` over orderings that fit inside `budget` moves."""
    k = plan.size
    values = customer_values(plan, objective)
    weights = customer_weights(plan)
    live = [i for i in range(k) if plan.reachable(i) and values[i] > 0]

    rate = [values[i] / weights[i] if weights[i] < INF else 0.0 for i in range(k)]
    by_rate = sorted(live, key=lambda i: -rate[i])
    by_weight = sorted(live, key=lambda i: weights[i])
    slot_of = {i: s for s, i in enumerate(by_rate)}
    slot_of_weight = {i: s for s, i in enumerate(by_weight)}

    available = AvailableSet([weights[i] for i in by_rate], [values[i] for i in by_rate])
    cheapest = CheapestTracker([weights[i] for i in by_weight])

    init_order, init_value, init_moves = build_initial_incumbent(plan, values, by_rate, live, budget)
    best = Solution(budget=budget, objective=objective)
    best.order, best.value, best.moves = init_order, init_value, init_moves

    memo: dict[tuple[int, int], float] = {}
    state = progress if progress is not None else {}
    state.setdefault("nodes", 0)
    started = time.perf_counter()

    def record(cur_order: list[int], value: float, travel: float) -> None:
        if value > best.value or (value == best.value and travel < best.moves):
            best.order = list(cur_order)
            best.value = value
            best.moves = travel

    def rec(prev: int, travel: float, value: float, mask: int, cur_order: list[int]) -> None:
        state["nodes"] += 1
        if state["nodes"] > node_limit:
            raise _Stop
        if time_limit is not None and state["nodes"] % 2048 == 0:
            if time.perf_counter() - started > time_limit:
                raise _Stop

        # Dominance cut: a no-cheaper arrival at this exact (customer, set of
        # served) state has already been explored, so anything reachable from
        # here has already been reached from there with at least as much
        # budget to spare.
        key = (prev, mask)
        prior = memo.get(key)
        if prior is not None and travel >= prior - 1e-9:
            return
        memo[key] = travel

        record(cur_order, value, travel)

        remaining = budget - travel
        if remaining <= 0:
            return
        if value + remaining * available.best_rate() <= best.value:
            return
        if value + available.knapsack_bound(remaining) <= best.value:
            return
        if value + tier3_bound(plan, mask, prev, values, live, remaining) <= best.value:
            return

        candidates = []
        for i in by_rate:
            if (mask >> i) & 1:
                continue
            step = plan.leg_cost(prev, i)
            if step == INF or travel + step > budget:
                continue
            candidates.append((values[i] / max(step, 1e-9), i, step))
        candidates.sort(key=lambda t: -t[0])

        for _, i, step in candidates:
            new_mask = mask | (1 << i)
            new_travel = travel + step
            new_value = value + values[i]
            slot_r, slot_w = slot_of[i], slot_of_weight[i]
            available.remove(slot_r)
            cheapest.remove(slot_w)
            cur_order.append(i)

            if new_travel + cheapest.min_weight() > budget:
                # Nothing left could fit even before descending: record this
                # leaf directly rather than pay for a call that would just
                # discover the same thing.
                record(cur_order, new_value, new_travel)
            else:
                rec(i, new_travel, new_value, new_mask, cur_order)

            cur_order.pop()
            available.restore(slot_r)
            cheapest.restore(slot_w)

    try:
        rec(-1, 0.0, 0.0, 0, [])
    except _Stop:
        best.optimal = False

    return _finish(best, plan, started, state["nodes"])


def brute_force(
    plan: RoutePlan,
    objective: Objective,
    budget: float,
    time_limit: float = 15.0,
    progress: dict | None = None,
) -> Solution:
    """Same search with no bounds at all: every feasible ordering is visited.

    The only cut is the budget itself (without it the space is infinite), so
    the node count is a fair picture of what the optimizations above save.
    """
    k = plan.size
    values = customer_values(plan, objective)
    live = [i for i in range(k) if plan.reachable(i) and values[i] > 0]

    used = [False] * k
    best = Solution(budget=budget, objective=objective, label="brute force")
    best.value = -1.0
    state = progress if progress is not None else {}
    state.setdefault("nodes", 0)
    started = time.perf_counter()

    def rec(prev: int, travel: float, value: float, order: list[int]) -> None:
        state["nodes"] += 1
        if state["nodes"] % 4096 == 0 and time.perf_counter() - started > time_limit:
            raise _Stop
        if value > best.value or (value == best.value and travel < best.moves):
            best.order = list(order)
            best.value = value
            best.moves = travel

        for i in live:
            if used[i]:
                continue
            step = plan.leg_cost(prev, i)
            if step == INF or travel + step > budget:
                continue
            used[i] = True
            order.append(i)
            rec(i, travel + step, value + values[i], order)
            order.pop()
            used[i] = False

    try:
        rec(-1, 0.0, 0.0, [])
    except _Stop:
        best.optimal = False

    return _finish(best, plan, started, state["nodes"])


def _finish(best: Solution, plan: RoutePlan, started: float, nodes: int) -> Solution:
    if best.value < 0:
        best.value = 0.0
        best.moves = 0.0
    best.nodes = nodes
    best.elapsed = time.perf_counter() - started
    best.legs = build_legs(plan, best.order)
    return best


def build_legs(plan: RoutePlan, order: list[int]) -> list[Leg]:
    legs: list[Leg] = []
    prev = -1
    for i in order:
        approach = plan.depot_path[i] if prev < 0 else plan.gap_path[prev][i]
        if approach:
            legs.append(Leg("approach", i, approach))
        delivery = plan.loaded_path[i]
        if delivery:
            legs.append(Leg("delivery", i, delivery))
        prev = i
    return legs
