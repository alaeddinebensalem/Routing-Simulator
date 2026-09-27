"""Grid model.

Cells are addressed as (row, col) with (0, 0) at the top-left.  Obstacles live
on the *edges* between cells, so blocking (r, c) EAST is exactly the same
obstacle as blocking (r, c + 1) WEST.  Border edges can be blocked too.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum

Cell = tuple[int, int]
EdgeKey = tuple[str, int, int]  # ("h" | "v", row, col) into the edge tables


class Side(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def delta(self) -> tuple[int, int]:
        """(d_row, d_col) step taken when crossing this side."""
        return _DELTAS[self]

    @property
    def opposite(self) -> "Side":
        return _OPPOSITES[self]


_DELTAS: dict[Side, tuple[int, int]] = {
    Side.NORTH: (-1, 0),
    Side.EAST: (0, 1),
    Side.SOUTH: (1, 0),
    Side.WEST: (0, -1),
}

_OPPOSITES: dict[Side, Side] = {
    Side.NORTH: Side.SOUTH,
    Side.EAST: Side.WEST,
    Side.SOUTH: Side.NORTH,
    Side.WEST: Side.EAST,
}


@dataclass
class Customer:
    """A shipment to be carried from `start` to `end`.

    `number` is the 1-based label drawn on the grid.
    """

    number: int
    start: Cell
    end: Cell


class Grid:
    def __init__(self, m: int, n: int) -> None:
        if m <= 0 or n <= 0:
            raise ValueError(f"grid size must be positive, got m={m}, n={n}")
        self.m = m
        self.n = n

        # h_edges[r][c]: horizontal edge above row r, spanning column c (r = 0..m)
        self.h_edges: list[list[bool]] = [[False] * n for _ in range(m + 1)]
        # v_edges[r][c]: vertical edge left of column c, spanning row r (c = 0..n)
        self.v_edges: list[list[bool]] = [[False] * (n + 1) for _ in range(m)]

        # Suggested fraction of edges to block; the side-panel slider sets it.
        self.obstacle_density: float = 0.2

        self.customers: list[Customer] = []

    # ---------------------------------------------------------------- edges
    @property
    def edge_count(self) -> int:
        return (self.m + 1) * self.n + self.m * (self.n + 1)

    @property
    def cell_count(self) -> int:
        return self.m * self.n

    @property
    def border_edge_count(self) -> int:
        return 2 * self.n + 2 * self.m

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.m and 0 <= col < self.n

    def edge_key(self, row: int, col: int, side: Side) -> EdgeKey:
        """Normalize (cell, side) to a slot in the edge tables."""
        side = Side(side)
        if side is Side.NORTH:
            return ("h", row, col)
        if side is Side.SOUTH:
            return ("h", row + 1, col)
        if side is Side.WEST:
            return ("v", row, col)
        return ("v", row, col + 1)

    def _table(self, key: EdgeKey) -> list[list[bool]]:
        return self.h_edges if key[0] == "h" else self.v_edges

    def is_obstacle(self, row: int, col: int, side: Side) -> bool:
        if not self.in_bounds(row, col):
            return True  # outside the grid is solid
        kind, r, c = self.edge_key(row, col, side)
        return self.h_edges[r][c] if kind == "h" else self.v_edges[r][c]

    def set_obstacle(self, row: int, col: int, side: Side, value: bool = True) -> None:
        if not self.in_bounds(row, col):
            raise IndexError(f"cell ({row}, {col}) is out of bounds")
        kind, r, c = self.edge_key(row, col, side)
        if kind == "h":
            self.h_edges[r][c] = value
        else:
            self.v_edges[r][c] = value

    def toggle_obstacle(self, row: int, col: int, side: Side) -> bool:
        new_value = not self.is_obstacle(row, col, side)
        self.set_obstacle(row, col, side, new_value)
        return new_value

    def clear_obstacles(self) -> None:
        for r in range(self.m + 1):
            for c in range(self.n):
                self.h_edges[r][c] = False
        for r in range(self.m):
            for c in range(self.n + 1):
                self.v_edges[r][c] = False

    def fill_obstacles(self) -> None:
        """Block every edge (the starting point for maze carving)."""
        for r in range(self.m + 1):
            for c in range(self.n):
                self.h_edges[r][c] = True
        for r in range(self.m):
            for c in range(self.n + 1):
                self.v_edges[r][c] = True

    def obstacle_count(self) -> int:
        return sum(sum(row) for row in self.h_edges) + sum(sum(row) for row in self.v_edges)

    def blocked_edges(self, interior_only: bool = False) -> list[EdgeKey]:
        keys: list[EdgeKey] = []
        for r in range(self.m + 1):
            if interior_only and (r == 0 or r == self.m):
                continue
            for c in range(self.n):
                if self.h_edges[r][c]:
                    keys.append(("h", r, c))
        for r in range(self.m):
            for c in range(self.n + 1):
                if interior_only and (c == 0 or c == self.n):
                    continue
                if self.v_edges[r][c]:
                    keys.append(("v", r, c))
        return keys

    def set_edge_key(self, key: EdgeKey, value: bool) -> None:
        kind, r, c = key
        if kind == "h":
            self.h_edges[r][c] = value
        else:
            self.v_edges[r][c] = value

    # ------------------------------------------------------------ movement
    def passable(self, row: int, col: int, side: Side) -> bool:
        """True if a robot may step out of (row, col) across `side`."""
        d_row, d_col = Side(side).delta
        nr, nc = row + d_row, col + d_col
        if not self.in_bounds(nr, nc):
            return False
        return not self.is_obstacle(row, col, side)

    def neighbors(self, cell: Cell):
        row, col = cell
        for side in (Side.NORTH, Side.EAST, Side.SOUTH, Side.WEST):
            if self.passable(row, col, side):
                d_row, d_col = side.delta
                yield (row + d_row, col + d_col)

    # ---------------------------------------------------- obstacle creation
    def create_obstacles(self) -> None:
        """Populate the grid with obstacles.  (Stub — implement me.)

        Obstacles sit on the edges *between* cells, so neighbouring cells share
        an edge: blocking the EAST side of (r, c) is the same wall as blocking
        the WEST side of (r, c + 1).  Build walls with, for example::

            self.set_obstacle(row, col, Side.EAST)
            self.set_obstacle(row, col, Side.NORTH, False)   # remove a wall

        `self.obstacle_density` holds the fraction of `self.edge_count` that
        the side-panel slider suggests blocking; honouring it is optional.

        Called once at startup and again whenever the "Regenerate" button (or
        the R key) is used while the generator is set to "Custom".  The grid is
        cleared of obstacles right before this runs.
        """
        pass

    # ------------------------------------------------------------ maze mode
    def perfect_maze_obstacle_count(self) -> int:
        """Walls left standing by a perfect maze (a spanning tree of cells)."""
        return self.edge_count - (self.cell_count - 1)

    def target_obstacle_count(self, density: float | None = None) -> int:
        """Desired wall count for `density`, capped at the perfect-maze count."""
        d = self.obstacle_density if density is None else density
        desired = int(round(max(0.0, min(1.0, d)) * self.edge_count))
        return min(desired, self.perfect_maze_obstacle_count())

    def carve_perfect_maze(self, rng: random.Random | None = None) -> None:
        """Block every edge, then carve a spanning tree with randomized DFS."""
        rng = rng or random
        self.fill_obstacles()

        start: Cell = (rng.randrange(self.m), rng.randrange(self.n))
        visited = {start}
        stack: list[Cell] = [start]
        while stack:
            row, col = stack[-1]
            options = []
            for side in (Side.NORTH, Side.EAST, Side.SOUTH, Side.WEST):
                d_row, d_col = side.delta
                nxt = (row + d_row, col + d_col)
                if self.in_bounds(*nxt) and nxt not in visited:
                    options.append((side, nxt))
            if not options:
                stack.pop()
                continue
            side, nxt = rng.choice(options)
            self.set_obstacle(row, col, side, False)
            visited.add(nxt)
            stack.append(nxt)

    def thin_walls(self, target: int, rng: random.Random | None = None) -> None:
        """Remove random interior walls until at most `target` remain."""
        rng = rng or random
        removable = self.blocked_edges(interior_only=True)
        rng.shuffle(removable)
        count = self.obstacle_count()
        while count > target and removable:
            self.set_edge_key(removable.pop(), False)
            count -= 1

    def generate_maze_obstacles(
        self, density: float | None = None, rng: random.Random | None = None
    ) -> None:
        """Perfect maze, then knock walls down until the density target is met.

        The request is capped at the perfect maze's own wall count: a perfect
        maze is the densest layout that still keeps every cell reachable.
        """
        rng = rng or random
        self.carve_perfect_maze(rng)
        self.thin_walls(self.target_obstacle_count(density), rng)

    # ----------------------------------------------------------- customers
    def marker_cells(self) -> dict[Cell, tuple[Customer, str]]:
        """Map of cell -> (customer, "start" | "end") for every marker drawn."""
        markers: dict[Cell, tuple[Customer, str]] = {}
        for customer in self.customers:
            markers[customer.start] = (customer, "start")
            markers[customer.end] = (customer, "end")
        return markers

    def customer_at(self, cell: Cell) -> Customer | None:
        for customer in self.customers:
            if customer.start == cell or customer.end == cell:
                return customer
        return None

    def random_free_cell(self, rng: random.Random | None = None) -> Cell:
        """A random cell, preferring one with no customer marker on it."""
        rng = rng or random
        taken = self.marker_cells()
        free = [
            (r, c)
            for r in range(self.m)
            for c in range(self.n)
            if (r, c) not in taken
        ]
        if free:
            return rng.choice(free)
        return (rng.randrange(self.m), rng.randrange(self.n))

    def add_customer(self, start: Cell, end: Cell) -> Customer:
        if not self.in_bounds(*start) or not self.in_bounds(*end):
            raise IndexError("customer cells must be inside the grid")
        if start == end:
            raise ValueError("a customer's start and end must differ")
        customer = Customer(len(self.customers) + 1, start, end)
        self.customers.append(customer)
        return customer

    def remove_customer(self, customer: Customer) -> None:
        self.customers.remove(customer)
        self._renumber()

    def clear_customers(self) -> None:
        self.customers.clear()

    def _renumber(self) -> None:
        for i, customer in enumerate(self.customers, start=1):
            customer.number = i

    def generate_customers(self, count: int, rng: random.Random | None = None) -> None:
        """Replace the customer list with `count` random start/end pairs.

        Every marker gets its own cell so the labels never overlap.
        """
        rng = rng or random
        self.clear_customers()
        count = max(0, min(count, self.cell_count // 2))
        cells = [(r, c) for r in range(self.m) for c in range(self.n)]
        rng.shuffle(cells)
        for i in range(count):
            self.add_customer(cells[2 * i], cells[2 * i + 1])
