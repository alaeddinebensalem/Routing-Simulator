"""Grid + side panel app.

Usage:  python main.py [m] [n]      (defaults: 20 rows, 25 columns)

Obstacles and customers are generated, not drawn by hand: everything is driven
from the side panel. Solving runs on a background thread so the window keeps
redrawing while it works.
"""

from __future__ import annotations

import queue
import random
import sys
import threading
import time

import pygame

import theme as T
import ui
from grid import Cell, Grid, Side
from solver import (
    Objective,
    RoutePlan,
    Solution,
    brute_force,
    random_tour_moves,
    solve,
)

INF = float("inf")
Edge = tuple[int, int, Side]

GENERATORS = ("Maze", "Custom")
MAX_CUSTOMERS = 100
BRUTE_TIME_LIMIT = 15.0


# ----------------------------------------------------------------------
# Pure geometry helpers
# ----------------------------------------------------------------------
def edge_segment(
    edge: Edge, origin: tuple[int, int], cell_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pixel endpoints of an edge, given as (row, col, side)."""
    row, col, side = edge
    ox, oy = origin
    if side is Side.NORTH or side is Side.SOUTH:
        r = row if side is Side.NORTH else row + 1
        y = oy + r * cell_size
        return (ox + col * cell_size, y), (ox + (col + 1) * cell_size, y)
    c = col if side is Side.WEST else col + 1
    x = ox + c * cell_size
    return (x, oy + row * cell_size), (x, oy + (row + 1) * cell_size)


def fit_cell_size(m: int, n: int) -> int:
    max_w, max_h = T.MAX_GRID_AREA
    size = min(max_w // n, max_h // m)
    return max(T.MIN_CELL_SIZE, min(T.MAX_CELL_SIZE, size))


# ----------------------------------------------------------------------
class App:
    def __init__(self, m: int, n: int) -> None:
        pygame.init()
        pygame.display.set_caption("Grid Router")

        self.grid = Grid(m, n)
        self.rng = random.Random()

        # display state
        self.theme_name = T.DEFAULT_THEME.name
        self.show_grid_lines = True
        self.generator = "Maze"
        self.customer_count = 6
        self._wall_surface: pygame.Surface | None = None
        self._walls_dirty = True
        self._route_layers: tuple[pygame.Surface, pygame.Surface] | None = None
        self._route_dirty = True

        # routing state
        self.depot: Cell = (0, 0)
        self.plan: RoutePlan | None = None
        self.plan_dirty = True
        self.budget_cap: float = 0.0
        self.budget_scale = 100.0
        self.objective = Objective.CUSTOMERS
        self.solution: Solution | None = None
        self.benchmark: tuple[Solution, Solution] | None = None
        self.leg_index = -1
        self.status = "Generate obstacles and customers, then solve."

        # background solving state
        self._solving = False
        self._solver_queue: queue.Queue | None = None
        self._solve_progress: dict = {"nodes": 0}
        self._solve_started = 0.0
        self._bench_fast_result: Solution | None = None

        # geometry
        self.cell_size = fit_cell_size(m, n)
        grid_w, grid_h = n * self.cell_size, m * self.cell_size
        panel_x = grid_w + 2 * T.PADDING
        self.window_w = panel_x + T.PANEL_WIDTH

        self.panel = ui.Panel(pygame.Rect(panel_x, 0, T.PANEL_WIDTH, 100))
        self._build_panel()

        self.window_h = min(
            max(grid_h + 2 * T.PADDING, self.panel.content_height), T.MAX_WINDOW_HEIGHT
        )
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        self.panel.set_rect(pygame.Rect(panel_x, 0, T.PANEL_WIDTH, self.window_h))
        self.origin = (T.PADDING, (self.window_h - grid_h) // 2)

        self.clock = pygame.time.Clock()
        self.running = True

        self.regenerate()
        self.generate_customers()

    # ------------------------------------------------------------ helpers
    @property
    def theme(self):
        return T.THEMES[self.theme_name]

    @property
    def wall_width(self) -> int:
        return max(2, min(T.WALL_WIDTH, self.cell_size // 3))

    @property
    def budget(self) -> float:
        return self.budget_cap * self.budget_scale / 100.0

    def cell_center(self, cell: Cell) -> tuple[int, int]:
        row, col = cell
        half = self.cell_size // 2
        return (col * self.cell_size + half, row * self.cell_size + half)

    def screen_center(self, cell: Cell) -> tuple[int, int]:
        x, y = self.cell_center(cell)
        return (self.origin[0] + x, self.origin[1] + y)

    def cell_rect(self, cell: Cell) -> pygame.Rect:
        row, col = cell
        ox, oy = self.origin
        return pygame.Rect(
            ox + col * self.cell_size, oy + row * self.cell_size, self.cell_size, self.cell_size
        )

    def invalidate(self) -> None:
        """Obstacles, customers or the robot start changed: routes are stale."""
        self.plan = None
        self.plan_dirty = True
        self.solution = None
        self.benchmark = None
        self.leg_index = -1
        self._walls_dirty = True
        self._route_dirty = True

    # --------------------------------------------------------- panel setup
    def _build_panel(self) -> None:
        p = self.panel
        g = self.grid

        p.add_title("Grid Controls")
        p.add_label(lambda: f"Size: {g.m} x {g.n}  ({g.cell_count} cells)", "muted")
        p.add_label(lambda: f"Obstacles: {g.obstacle_count()} / {g.edge_count}")

        p.add_section("Generation")
        p.add_slider(
            "Obstacle density",
            0.0, 100.0, g.obstacle_density * 100.0,
            self._set_density,
            step=1.0,
            fmt=lambda v: f"{v:.0f}%",
        )
        p.add_label(lambda: f"Maze cap: {g.perfect_maze_obstacle_count()} walls", "muted")
        p.add_button(lambda: f"Generator: {self.generator}", self._cycle_generator)
        p.add_row([
            ui.Button("Regenerate (R)", self.regenerate),
            ui.Button("Clear (C)", self._clear_obstacles),
        ])

        p.add_section("Customers")
        p.add_slider(
            "Customers",
            1.0, float(max(1, min(MAX_CUSTOMERS, g.cell_count // 2))), float(self.customer_count),
            self._set_customer_count,
            step=1.0,
            fmt=lambda v: f"{int(v)}",
        )
        p.add_row([
            ui.Button("Generate (G)", self.generate_customers),
            ui.Button("Clear (X)", self._clear_customers),
        ])
        p.add_label(lambda: f"Customers: {len(g.customers)}", "muted")
        p.add_row([
            ui.Button(lambda: f"Robot start: {self.depot}", self.reroll_depot),
        ])

        p.add_section("Routing")
        p.add_button(lambda: f"Maximize: {self.objective.value}", self._cycle_objective)
        p.add_slider(
            "Move budget",
            10.0, 100.0, 100.0,
            self._set_budget_scale,
            step=5.0,
            fmt=lambda v: f"{v:.0f}%",
        )
        p.add_label(self._budget_text, "muted")
        p.add_row([
            ui.Button("Reroll budget", self._reroll_budget),
            ui.Button("Solve (S)", self.run_solver),
        ])
        p.add_label(self._solving_text, "muted")
        p.add_label(self._result_text)
        p.add_label(self._result_detail, "muted")

        p.add_section("Benchmark")
        p.add_button("Compare vs brute force (B)", self.run_benchmark)
        p.add_label(self._bench_fast, "muted")
        p.add_label(self._bench_slow, "muted")
        p.add_label(self._bench_verdict, "muted")

        p.add_section("Path playback")
        p.add_row([
            ui.Button("< Prev", lambda: self.step_leg(-1)),
            ui.Button("Next >", lambda: self.step_leg(1)),
        ])
        p.add_label(self._leg_text, "muted")

        p.add_section("Display")
        p.add_toggle("Show grid lines", self.show_grid_lines, self._set_grid_lines)
        p.add_button(lambda: f"Theme: {self.theme_name}", self._cycle_theme)

        p.add_separator()
        p.add_label(self._status_text, "muted")
        p.add_button("Quit (Esc)", self.quit)

    # ------------------------------------------------------- label sources
    def _status_text(self) -> str:
        return self.status

    def _budget_text(self) -> str:
        if self.budget_cap <= 0:
            return "Budget: solve to compute the cap"
        return f"Budget: {self.budget:.0f} of {self.budget_cap:.0f} moves"

    def _solving_text(self) -> str:
        if not self._solving:
            return ""
        elapsed = time.perf_counter() - self._solve_started
        nodes = self._solve_progress.get("nodes", 0)
        phase = self._solve_progress.get("phase", "bnb")
        label = "branch and bound" if phase == "bnb" else "brute force"
        return f"Solving ({label})... {nodes:,} nodes, {elapsed:.1f}s"

    def _result_text(self) -> str:
        s = self.solution
        if s is None:
            return "No route yet."
        return f"Served {len(s.order)}/{len(self.grid.customers)}  |  {s.moves:.0f} moves"

    def _result_detail(self) -> str:
        s = self.solution
        if s is None or self.plan is None:
            return ""
        loaded = sum(self.plan.loaded_dist[i] for i in s.order)
        flag = "optimal" if s.optimal else "capped"
        return f"Loaded: {loaded:.0f}  |  {flag}  |  {s.nodes:,} nodes"

    def _bench_fast(self) -> str:
        if self.benchmark is None:
            return "No benchmark yet."
        fast = self.benchmark[0]
        return f"B&B:   {fast.value:.0f} in {fast.nodes:,} nodes ({fast.elapsed:.2f}s)"

    def _bench_slow(self) -> str:
        if self.benchmark is None:
            return ""
        slow = self.benchmark[1]
        tail = "" if slow.optimal else " timeout"
        return f"Brute: {slow.value:.0f} in {slow.nodes:,} nodes ({slow.elapsed:.1f}s{tail})"

    def _bench_verdict(self) -> str:
        if self.benchmark is None:
            return ""
        fast, slow = self.benchmark
        ratio = slow.nodes / max(fast.nodes, 1)
        if not slow.optimal:
            return f"{ratio:,.0f}x fewer nodes (brute force unfinished)"
        agree = "same profit" if fast.value == slow.value else "PROFIT MISMATCH"
        return f"{ratio:,.0f}x fewer nodes  |  {agree}"

    def _leg_text(self) -> str:
        s = self.solution
        if s is None or not s.legs:
            return "No legs to show."
        if self.leg_index < 0:
            return f"Leg 0 / {len(s.legs)} (hidden)"
        leg = s.legs[self.leg_index]
        number = self.grid.customers[leg.customer].number if leg.customer is not None else "?"
        what = (
            f"drive to customer {number}"
            if leg.kind == "approach"
            else f"deliver customer {number}"
        )
        return f"Leg {self.leg_index + 1} / {len(s.legs)}: {what} ({leg.moves} moves)"

    # ---------------------------------------------------------- callbacks
    def _set_density(self, value: float) -> None:
        self.grid.obstacle_density = value / 100.0

    def _set_grid_lines(self, value: bool) -> None:
        self.show_grid_lines = value

    def _set_customer_count(self, value: float) -> None:
        self.customer_count = int(value)

    def _set_budget_scale(self, value: float) -> None:
        self.budget_scale = value
        self.solution = None
        self.benchmark = None
        self.leg_index = -1
        self._route_dirty = True

    def _cycle_generator(self) -> None:
        if self._busy_guard():
            return
        i = GENERATORS.index(self.generator)
        self.generator = GENERATORS[(i + 1) % len(GENERATORS)]

    def _cycle_theme(self) -> None:
        names = list(T.THEMES)
        self.theme_name = names[(names.index(self.theme_name) + 1) % len(names)]
        self._walls_dirty = True
        self._route_dirty = True

    def _cycle_objective(self) -> None:
        if self._busy_guard():
            return
        members = list(Objective)
        i = members.index(self.objective)
        self.objective = members[(i + 1) % len(members)]
        self.solution = None
        self.benchmark = None
        self.leg_index = -1
        self._route_dirty = True

    def _clear_obstacles(self) -> None:
        if self._busy_guard():
            return
        self.grid.clear_obstacles()
        self.invalidate()
        self.status = "Obstacles cleared."

    def _clear_customers(self) -> None:
        if self._busy_guard():
            return
        self.grid.clear_customers()
        self.invalidate()
        self.budget_cap = 0.0
        self.status = "Customers cleared."

    def generate_customers(self) -> None:
        if self._busy_guard():
            return
        self.grid.generate_customers(self.customer_count, self.rng)
        self.depot = self.grid.random_free_cell(self.rng)
        self.invalidate()
        self.budget_cap = 0.0
        self.status = f"{len(self.grid.customers)} customers, robot at {self.depot}."

    def reroll_depot(self) -> None:
        if self._busy_guard():
            return
        self.depot = self.grid.random_free_cell(self.rng)
        self.invalidate()
        self.budget_cap = 0.0
        self.status = f"Robot starts at {self.depot}."

    def regenerate(self) -> None:
        """R: rebuild obstacles with the selected generator."""
        if self._busy_guard():
            return
        self.grid.clear_obstacles()
        if self.generator == "Maze":
            self.grid.generate_maze_obstacles(rng=self.rng)
            self.status = "Maze generated."
        else:
            self.grid.create_obstacles()
            self.status = "create_obstacles() called."
        self.depot = self.grid.random_free_cell(self.rng)
        self.invalidate()
        self.budget_cap = 0.0

    # ------------------------------------------------------------ routing
    def _busy_guard(self) -> bool:
        """True (and sets a status message) if a solve is already running."""
        if self._solving:
            self.status = "Still solving - please wait."
            return True
        return False

    def _announce(self, message: str) -> None:
        """Show a status line immediately (building the plan can take a while)."""
        self.status = message
        self.draw()
        pygame.display.flip()

    def ensure_plan(self) -> RoutePlan | None:
        if not self.grid.customers:
            return None
        if self.plan is None or self.plan_dirty:
            self._announce("Running A* over all customer pairs...")
            self.plan = RoutePlan.build(self.grid, self.depot)
            self.plan_dirty = False
            self.budget_cap = random_tour_moves(self.plan, self.rng)
            if self.budget_cap == INF:
                self.budget_cap = 0.0
        return self.plan

    def _reroll_budget(self) -> None:
        if self._busy_guard():
            return
        plan = self.ensure_plan()
        if plan is None:
            self.status = "Generate customers first."
            return
        cap = random_tour_moves(plan, self.rng)
        self.budget_cap = 0.0 if cap == INF else cap
        self.solution = None
        self.benchmark = None
        self.leg_index = -1
        self._route_dirty = True
        self.status = "New random tour sets the move cap."

    def run_solver(self) -> None:
        if self._busy_guard():
            return
        plan = self.ensure_plan()
        if plan is None:
            self.status = "Generate customers first."
            return
        if self.budget_cap <= 0:
            self.status = "No reachable customers."
            return
        self._start_solver_thread("solve")

    def run_benchmark(self) -> None:
        """Solve twice - with and without bounds - and compare the search cost."""
        if self._busy_guard():
            return
        plan = self.ensure_plan()
        if plan is None:
            self.status = "Generate customers first."
            return
        if self.budget_cap <= 0:
            self.status = "No reachable customers."
            return
        self._start_solver_thread("benchmark")

    def _start_solver_thread(self, kind: str) -> None:
        """Run solve()/brute_force() on a worker thread so the window keeps
        redrawing (and the panel keeps showing live node counts) while it
        works, instead of freezing on a hard time cap."""
        self._solve_progress = {"nodes": 0, "phase": "bnb"}
        self._solve_started = time.perf_counter()
        self._solving = True
        self._bench_fast_result = None
        self.status = "Solving in the background..."

        plan, objective, budget, progress = self.plan, self.objective, self.budget, self._solve_progress
        q: queue.Queue = queue.Queue()
        self._solver_queue = q

        def work() -> None:
            try:
                if kind == "solve":
                    result = solve(plan, objective, budget, progress=progress)
                    q.put(("solve", result))
                else:
                    fast = solve(plan, objective, budget, progress=progress)
                    q.put(("bench_fast", fast))
                    progress["nodes"] = 0
                    progress["phase"] = "brute"
                    slow = brute_force(plan, objective, budget, BRUTE_TIME_LIMIT, progress=progress)
                    q.put(("bench_slow", slow))
            except Exception as exc:  # keep the UI alive even if solving blows up
                q.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _poll_solver(self) -> None:
        """Drain any results the background solver thread has posted."""
        if self._solver_queue is None:
            return
        try:
            while True:
                kind, payload = self._solver_queue.get_nowait()
                if kind == "solve":
                    self.solution = payload
                    self.benchmark = None
                    self.leg_index = 0 if payload.legs else -1
                    self._route_dirty = True
                    self._solving = False
                    self.status = "Route ready - use the arrows."
                elif kind == "bench_fast":
                    self._bench_fast_result = payload
                    self.solution = payload
                    self.leg_index = 0 if payload.legs else -1
                    self._route_dirty = True
                    self.status = f"Benchmark: brute force running (up to {BRUTE_TIME_LIMIT:.0f}s)..."
                elif kind == "bench_slow":
                    self.benchmark = (self._bench_fast_result, payload)
                    self._solving = False
                    self.status = (
                        "Brute force hit the time limit."
                        if not payload.optimal
                        else "Benchmark done - the grid shows the optimized route."
                    )
                elif kind == "error":
                    self._solving = False
                    self.status = f"Solve failed: {payload}"
        except queue.Empty:
            pass

    def step_leg(self, delta: int) -> None:
        if self.solution is None or not self.solution.legs:
            return
        self.leg_index = max(-1, min(len(self.solution.legs) - 1, self.leg_index + delta))
        self._route_dirty = True


    # ------------------------------------------------------------- events
    def quit(self) -> None:
        self.running = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if self.panel.handle_event(event):
            return
        if event.type == pygame.KEYDOWN:
            self._handle_key(event)

    def _handle_key(self, event: pygame.event.Event) -> None:
        key = event.key
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
        elif key == pygame.K_r:
            self.regenerate()
        elif key == pygame.K_c:
            self._clear_obstacles()
        elif key == pygame.K_x:
            self._clear_customers()
        elif key == pygame.K_g:
            self.generate_customers()
        elif key == pygame.K_s:
            self.run_solver()
        elif key == pygame.K_b:
            self.run_benchmark()
        elif key == pygame.K_LEFT:
            self.step_leg(-1)
        elif key == pygame.K_RIGHT:
            self.step_leg(1)

    # ------------------------------------------------------------ drawing
    def draw(self) -> None:
        theme = self.theme
        self.screen.fill(theme.window_bg)
        grid_rect = pygame.Rect(
            self.origin[0], self.origin[1],
            self.grid.n * self.cell_size, self.grid.m * self.cell_size,
        )
        pygame.draw.rect(self.screen, theme.cell_bg, grid_rect)

        self._draw_markers(theme)
        if self.show_grid_lines:
            self._draw_grid_lines(theme, grid_rect)
        self._draw_route(theme)
        self._draw_depot(theme)
        self._draw_walls(theme)

        self.panel.draw(self.screen, theme)

    def _draw_grid_lines(self, theme, grid_rect: pygame.Rect) -> None:
        cs = self.cell_size
        ox, oy = self.origin
        for c in range(self.grid.n + 1):
            x = ox + c * cs
            pygame.draw.line(
                self.screen, theme.grid_line, (x, grid_rect.y), (x, grid_rect.bottom),
                T.GRID_LINE_WIDTH,
            )
        for r in range(self.grid.m + 1):
            y = oy + r * cs
            pygame.draw.line(
                self.screen, theme.grid_line, (grid_rect.x, y), (grid_rect.right, y),
                T.GRID_LINE_WIDTH,
            )

    def _draw_walls(self, theme) -> None:
        """Walls are static between edits, so they live on a cached layer."""
        width = self.wall_width
        margin = width
        if self._walls_dirty or self._wall_surface is None:
            cs = self.cell_size
            size = (self.grid.n * cs + 2 * margin, self.grid.m * cs + 2 * margin)
            surface = pygame.Surface(size, pygame.SRCALPHA)
            radius = max(1, width // 2)
            color = theme.wall
            offset = (margin, margin)
            for r in range(self.grid.m + 1):
                for c in range(self.grid.n):
                    if self.grid.h_edges[r][c]:
                        edge = (r, c, Side.NORTH) if r < self.grid.m else (r - 1, c, Side.SOUTH)
                        p1, p2 = edge_segment(edge, offset, cs)
                        self._stroke(surface, color, p1, p2, width, radius)
            for r in range(self.grid.m):
                for c in range(self.grid.n + 1):
                    if self.grid.v_edges[r][c]:
                        edge = (r, c, Side.WEST) if c < self.grid.n else (r, c - 1, Side.EAST)
                        p1, p2 = edge_segment(edge, offset, cs)
                        self._stroke(surface, color, p1, p2, width, radius)
            self._wall_surface = surface
            self._walls_dirty = False
        self.screen.blit(self._wall_surface, (self.origin[0] - margin, self.origin[1] - margin))

    @staticmethod
    def _stroke(surface, color, p1, p2, width: int, radius: int) -> None:
        pygame.draw.line(surface, color, p1, p2, width)
        pygame.draw.circle(surface, color, p1, radius)
        pygame.draw.circle(surface, color, p2, radius)

    def _draw_markers(self, theme) -> None:
        total = len(self.grid.customers)
        font = T.get_font(max(9, int(self.cell_size * 0.5)), True)
        for i, customer in enumerate(self.grid.customers):
            for cell, kind in ((customer.start, "start"), (customer.end, "end")):
                color = (
                    T.customer_start_color(i, total, theme)
                    if kind == "start"
                    else T.customer_end_color(i, total, theme)
                )
                rect = self.cell_rect(cell).inflate(-2, -2)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
                image = font.render(str(customer.number), True, theme.marker_text)
                self.screen.blit(image, image.get_rect(center=rect.center))

    def _draw_depot(self, theme) -> None:
        center = self.screen_center(self.depot)
        radius = max(4, self.cell_size // 4)
        pygame.draw.circle(self.screen, theme.accent, center, radius)
        pygame.draw.circle(self.screen, theme.window_bg, center, max(2, radius // 2))

    def _leg_color(self, leg, theme):
        if leg.kind == "approach":
            return theme.path_approach
        return T.delivery_path_color(leg.customer or 0, theme)

    def _draw_route(self, theme) -> None:
        """Earlier legs are drawn faint and thin; the focused leg fills the cell."""
        solution = self.solution
        if solution is None or self.leg_index < 0 or not solution.legs:
            return

        if self._route_dirty or self._route_layers is None:
            size = (self.grid.n * self.cell_size, self.grid.m * self.cell_size)
            faded = pygame.Surface(size, pygame.SRCALPHA)
            focused = pygame.Surface(size, pygame.SRCALPHA)
            thin = max(2, self.cell_size // 4)
            thick = max(3, self.cell_size // 2)

            for index in range(self.leg_index):
                leg = solution.legs[index]
                self._stroke_path(faded, self._leg_color(leg, theme), leg.path, thin)
            current = solution.legs[self.leg_index]
            self._stroke_path(focused, self._leg_color(current, theme), current.path, thick)

            # Opaque strokes plus one surface-wide alpha: overlapping segments
            # inside a leg stay an even tone instead of piling up.
            faded.set_alpha(T.PATH_FADED_ALPHA)
            focused.set_alpha(T.PATH_FOCUSED_ALPHA)
            self._route_layers = (faded, focused)
            self._route_dirty = False

        for layer in self._route_layers:
            self.screen.blit(layer, self.origin)

    def _stroke_path(self, surface, color, path, width: int) -> None:
        points = [self.cell_center(cell) for cell in path]
        radius = max(1, width // 2)
        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, width)
        for point in points:
            pygame.draw.circle(surface, color, point, radius)

    # --------------------------------------------------------------- loop
    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self._poll_solver()
            self.draw()
            pygame.display.flip()
            self.clock.tick(T.FPS)
        pygame.quit()


def parse_args(argv: list[str]) -> tuple[int, int]:
    m, n = 40, 50
    if len(argv) >= 2:
        m = int(argv[1])
    if len(argv) >= 3:
        n = int(argv[2])
    return m, n


def main() -> None:
    try:
        m, n = parse_args(sys.argv)
    except ValueError:
        print("usage: python main.py [rows] [cols]")
        raise SystemExit(2)
    App(m, n).run()


if __name__ == "__main__":
    main()
