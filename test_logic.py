"""Headless sanity checks for the pygame-free modules."""
from __future__ import annotations

import itertools
import random

from grid import Grid, Side
from pathfinding import astar, path_moves
from solver import (
    AvailableSet,
    CheapestTracker,
    Objective,
    RoutePlan,
    _scan_knapsack_bound,
    brute_force,
    customer_values,
    customer_weights,
    hungarian_min_cost,
    random_tour_moves,
    solve,
    tier3_bound,
)


def check_edges():
    g = Grid(3, 4)
    assert g.edge_count == 4 * 4 + 3 * 5 == 31
    g.set_obstacle(1, 1, Side.EAST)
    assert g.is_obstacle(1, 2, Side.WEST)
    assert g.obstacle_count() == 1
    g.toggle_obstacle(1, 2, Side.WEST)
    assert g.obstacle_count() == 0
    try:
        Grid(0, 3)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        g.set_obstacle(5, 5, Side.NORTH)
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
    assert Side.NORTH.delta == (-1, 0) and Side.NORTH.opposite is Side.SOUTH
    print("edges ok")


def reachable_all(g: Grid) -> bool:
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        for nb in g.neighbors(stack.pop()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == g.cell_count


def check_maze():
    rng = random.Random(7)
    for m, n in [(1, 1), (1, 6), (6, 1), (5, 7), (15, 20)]:
        g = Grid(m, n)
        g.carve_perfect_maze(rng)
        assert g.obstacle_count() == g.perfect_maze_obstacle_count(), (m, n)
        assert reachable_all(g)
        for density in (0.0, 0.1, 0.35, 1.0):
            g.generate_maze_obstacles(density, rng)
            target = g.target_obstacle_count(density)
            assert g.obstacle_count() <= max(target, g.border_edge_count)
            assert g.obstacle_count() >= min(target, g.border_edge_count)
            assert reachable_all(g)
            assert all(g.h_edges[0]) and all(g.h_edges[m])
    print("maze ok")


def check_astar():
    g = Grid(5, 5)
    g.generate_maze_obstacles(0.25, random.Random(3))
    for a, b in [((0, 0), (4, 4)), ((2, 3), (0, 1))]:
        path = astar(g, a, b)
        assert path and path[0] == a and path[-1] == b
        for p, q in zip(path, path[1:]):
            assert abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1
    g2 = Grid(3, 3)
    for side in Side:
        g2.set_obstacle(1, 1, side)
    assert astar(g2, (0, 0), (1, 1)) is None
    assert path_moves(None) == float("inf")
    print("astar ok")


def check_hungarian():
    """hungarian_min_cost must match brute-force assignment exactly."""
    import itertools

    rng = random.Random(1)

    def brute(cost):
        n, m = len(cost), len(cost[0])
        best = float("inf")
        for cols in itertools.permutations(range(m), n):
            total = sum(cost[i][cols[i]] for i in range(n))
            best = min(best, total)
        return best

    for _ in range(200):
        n = rng.randint(0, 5)
        m = rng.randint(n, n + 3) if n else rng.randint(0, 3)
        cost = [[rng.uniform(0, 50) for _ in range(m)] for _ in range(n)]
        got, assignment = hungarian_min_cost(cost)
        if n == 0:
            assert got == 0.0 and assignment == []
            continue
        assert len(set(assignment)) == n
        recomputed = sum(cost[i][assignment[i]] for i in range(n))
        assert abs(recomputed - got) < 1e-6
        assert abs(got - brute(cost)) < 1e-6
    print("hungarian ok (matches brute-force assignment)")



def check_available_set():
    """The Fenwick bounds must agree with the plain O(n) scan, always."""
    rng = random.Random(21)
    for _ in range(300):
        n = rng.randint(0, 9)
        weights = [rng.uniform(1, 40) for _ in range(n)]
        values = [rng.uniform(1, 60) for _ in range(n)]
        order = sorted(range(n), key=lambda i: -(values[i] / weights[i]))
        avail = AvailableSet([weights[i] for i in order], [values[i] for i in order])
        used = [False] * n
        slot_of = {i: s for s, i in enumerate(order)}
        for _ in range(12):
            capacity = rng.uniform(0, 200)
            want = _scan_knapsack_bound(order, used, weights, values, capacity)
            got = avail.knapsack_bound(capacity)
            assert abs(want - got) < 1e-6, (want, got, used)
            rates = [values[i] / weights[i] for i in order if not used[i]]
            assert abs(avail.best_rate() - (rates[0] if rates else 0.0)) < 1e-9
            free = [i for i in range(n) if not used[i]]
            if free and rng.random() < 0.6:
                i = rng.choice(free)
                used[i] = True
                avail.remove(slot_of[i])
            else:
                taken = [i for i in range(n) if used[i]]
                if taken:
                    i = rng.choice(taken)
                    used[i] = False
                    avail.restore(slot_of[i])
    print("available set ok (Fenwick == scan)")


def check_cheapest_tracker():
    """CheapestTracker's min_weight must match a plain scan under remove/restore."""
    rng = random.Random(13)
    for _ in range(300):
        n = rng.randint(0, 9)
        weights = sorted(rng.uniform(1, 40) for _ in range(n))
        tracker = CheapestTracker(weights)
        used = [False] * n
        for _ in range(12):
            want = min((weights[i] for i in range(n) if not used[i]), default=float("inf"))
            got = tracker.min_weight()
            if want == float("inf"):
                assert got == float("inf")
            else:
                assert abs(want - got) < 1e-9, (want, got)
            free = [i for i in range(n) if not used[i]]
            if free and rng.random() < 0.6:
                i = rng.choice(free)
                used[i] = True
                tracker.remove(i)
            else:
                taken = [i for i in range(n) if used[i]]
                if taken:
                    i = rng.choice(taken)
                    used[i] = False
                    tracker.restore(i)
    print("cheapest tracker ok (min_weight == scan)")




def check_weights_admissible():
    """weight[i] must never exceed the travel actually spent serving i."""
    rng = random.Random(4)
    for _ in range(8):
        g = Grid(6, 7)
        g.generate_maze_obstacles(0.25, rng)
        g.generate_customers(rng.randint(2, 5), rng)
        plan = RoutePlan.build(g, g.random_free_cell(rng))
        weights = customer_weights(plan)
        for i in range(plan.size):
            if not plan.reachable(i):
                continue
            assert weights[i] <= plan.leg_cost(-1, i) + 1e-9
            for j in range(plan.size):
                if j != i and plan.reachable(j):
                    assert weights[i] <= plan.leg_cost(j, i) + 1e-9
    print("weights ok (never overcharge a leg)")


def enumerate_best(plan, objective, budget):
    values = customer_values(plan, objective)
    best = 0.0
    idx = list(range(plan.size))
    for r in range(len(idx) + 1):
        for perm in itertools.permutations(idx, r):
            cost, prev, val = 0.0, -1, 0.0
            ok = True
            for i in perm:
                cost += plan.leg_cost(prev, i)
                if cost > budget:
                    ok = False
                    break
                val += values[i]
                prev = i
            if ok and val > best:
                best = val
    return best


def check_solver():
    rng = random.Random(11)
    for trial in range(12):
        g = Grid(6, 6)
        g.generate_maze_obstacles(rng.choice([0.05, 0.2, 0.4]), rng)
        g.generate_customers(rng.randint(2, 5), rng)
        plan = RoutePlan.build(g, g.random_free_cell(rng))
        cap = random_tour_moves(plan, rng)
        for scale in (0.3, 0.6, 1.0):
            budget = cap * scale
            for objective in Objective:
                got = solve(plan, objective, budget)
                want = enumerate_best(plan, objective, budget)
                brute = brute_force(plan, objective, budget, 30.0)
                assert got.value == want, (trial, scale, objective, got.value, want)
                assert brute.value == want, (brute.value, want)
                assert got.moves <= budget + 1e-9
                assert got.nodes <= brute.nodes
                assert len(got.legs) == 2 * len(got.order)
                assert sum(leg.moves for leg in got.legs) == got.moves
    print("solver ok (bounds, brute force and permutations all agree)")


def check_full_tour():
    rng = random.Random(5)
    g = Grid(10, 10)
    g.generate_maze_obstacles(0.3, rng)
    g.generate_customers(7, rng)
    plan = RoutePlan.build(g, g.random_free_cell(rng))
    cap = random_tour_moves(plan, rng)
    result = solve(plan, Objective.CUSTOMERS, cap)
    assert result.value == 7, result.value
    assert result.optimal
    print("full tour ok:", result.moves, "of", cap, "moves,", result.nodes, "nodes")


def bench():
    rng = random.Random(2)
    header = (
        f"{'k':>3} {'budget':>7} {'objective':<14} {'B&B nodes':>11} "
        f"{'brute nodes':>12} {'B&B s':>7} {'brute s':>8}"
    )
    print(header)
    for k in (8, 10, 12, 16, 20):
        g = Grid(20, 25)
        g.generate_maze_obstacles(0.3, rng)
        g.generate_customers(k, rng)
        plan = RoutePlan.build(g, g.random_free_cell(rng))
        cap = random_tour_moves(plan, rng)
        for scale in (0.5, 1.0):
            for objective in Objective:
                fast = solve(plan, objective, cap * scale)
                slow = brute_force(plan, objective, cap * scale, 15.0)
                mark = "" if slow.optimal else " (timeout)"
                print(
                    f"{k:>3} {scale:>6.0%} {objective.name:<14} {fast.nodes:>11,} "
                    f"{slow.nodes:>12,} {fast.elapsed:>7.2f} {slow.elapsed:>8.2f}{mark}"
                )


if __name__ == "__main__":
    check_edges()
    check_maze()
    check_astar()
    check_hungarian()
    check_available_set()
    check_cheapest_tracker()
    check_weights_admissible()
    check_solver()
    check_full_tour()
    print()
    bench()
    print("\nALL OK")
