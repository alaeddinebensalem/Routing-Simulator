"""A* over the grid graph, using Manhattan distance as the heuristic.

Every move costs 1 and moves are 4-directional, so Manhattan distance is
admissible and consistent.
"""

from __future__ import annotations

import heapq
from itertools import count

from grid import Cell, Grid


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest cell path from `start` to `goal`, or None if unreachable."""
    if not grid.in_bounds(*start) or not grid.in_bounds(*goal):
        return None
    if start == goal:
        return [start]

    tie = count()
    open_heap: list[tuple[int, int, Cell]] = [(manhattan(start, goal), next(tie), start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, int] = {start: 0}
    closed: set[Cell] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed.add(current)

        tentative = g_score[current] + 1
        for neighbor in grid.neighbors(current):
            if neighbor in closed:
                continue
            if tentative < g_score.get(neighbor, 1 << 30):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(
                    open_heap, (tentative + manhattan(neighbor, goal), next(tie), neighbor)
                )

    return None


def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_moves(path: list[Cell] | None) -> int | float:
    """Number of moves in a path (INF when there is no path)."""
    if path is None:
        return float("inf")
    return len(path) - 1
