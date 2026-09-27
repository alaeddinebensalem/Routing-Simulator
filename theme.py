"""All look-and-feel for the grid app: layout metrics, fonts and color themes.

Adding a new theme only requires creating a `Theme` instance and registering it
in `THEMES`.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from functools import lru_cache

import pygame

RGB = tuple[int, int, int]

# --------------------------------------------------------------------------
# Layout metrics
# --------------------------------------------------------------------------
PANEL_WIDTH = 320
PADDING = 16
WIDGET_GAP = 8
SECTION_GAP = 14

BUTTON_HEIGHT = 30
SLIDER_HEIGHT = 40
TOGGLE_HEIGHT = 26
SEPARATOR_HEIGHT = 9
CORNER_RADIUS = 6

GRID_LINE_WIDTH = 1
WALL_WIDTH = 7
EDGE_PICK_TOLERANCE = 12

MIN_CELL_SIZE = 16
MAX_CELL_SIZE = 64
MAX_GRID_AREA = (920, 760)
MAX_WINDOW_HEIGHT = 840

PATH_WIDTH = 4
PATH_FADED_ALPHA = 60      # legs already walked
PATH_FOCUSED_ALPHA = 215   # the leg the arrows are on
FPS = 60

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
FONT_NAME = "dejavusans,verdana,arial,helvetica"
FONT_SIZE_TITLE = 21
FONT_SIZE_SECTION = 15
FONT_SIZE_BODY = 14
FONT_SIZE_SMALL = 12


@lru_cache(maxsize=64)
def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Cached SysFont lookup (font creation is expensive)."""
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.SysFont(FONT_NAME, size, bold=bold)


# --------------------------------------------------------------------------
# Themes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    name: str

    window_bg: RGB
    cell_bg: RGB
    grid_line: RGB
    wall: RGB
    edge_hover: RGB

    panel_bg: RGB
    panel_border: RGB

    text: RGB
    text_muted: RGB
    title: RGB
    accent: RGB

    button_bg: RGB
    button_hover: RGB
    button_press: RGB
    button_text: RGB

    slider_track: RGB
    slider_knob: RGB
    toggle_off: RGB

    # Customer markers and route overlay.
    start_shade_lo: RGB
    start_shade_hi: RGB
    end_shade_lo: RGB
    end_shade_hi: RGB
    marker_text: RGB
    path_approach: RGB
    path_value: float  # brightness used for generated delivery-path hues


DARK = Theme(
    name="Dark",
    window_bg=(18, 20, 25),
    cell_bg=(32, 36, 44),
    grid_line=(48, 54, 64),
    wall=(226, 232, 240),
    edge_hover=(250, 204, 21),
    panel_bg=(25, 28, 35),
    panel_border=(48, 54, 64),
    text=(226, 232, 240),
    text_muted=(140, 150, 165),
    title=(248, 250, 252),
    accent=(56, 189, 248),
    button_bg=(45, 51, 62),
    button_hover=(60, 68, 82),
    button_press=(80, 90, 106),
    button_text=(226, 232, 240),
    slider_track=(45, 51, 62),
    slider_knob=(56, 189, 248),
    toggle_off=(58, 64, 76),
    start_shade_lo=(96, 100, 108),
    start_shade_hi=(205, 210, 218),
    end_shade_lo=(126, 29, 29),
    end_shade_hi=(248, 113, 113),
    marker_text=(12, 14, 18),
    path_approach=(150, 158, 172),
    path_value=0.95,
)

LIGHT = Theme(
    name="Light",
    window_bg=(238, 240, 244),
    cell_bg=(252, 252, 253),
    grid_line=(214, 218, 226),
    wall=(30, 38, 52),
    edge_hover=(202, 138, 4),
    panel_bg=(248, 249, 251),
    panel_border=(210, 215, 224),
    text=(28, 33, 42),
    text_muted=(105, 114, 130),
    title=(15, 20, 28),
    accent=(2, 132, 199),
    button_bg=(226, 230, 238),
    button_hover=(212, 218, 228),
    button_press=(196, 203, 216),
    button_text=(28, 33, 42),
    slider_track=(214, 219, 228),
    slider_knob=(2, 132, 199),
    toggle_off=(200, 206, 216),
    start_shade_lo=(215, 218, 224),
    start_shade_hi=(128, 134, 145),
    end_shade_lo=(254, 202, 202),
    end_shade_hi=(185, 28, 28),
    marker_text=(255, 255, 255),
    path_approach=(120, 128, 142),
    path_value=0.72,
)

THEMES: dict[str, Theme] = {DARK.name: DARK, LIGHT.name: LIGHT}
DEFAULT_THEME = DARK


# --------------------------------------------------------------------------
# Color helpers
# --------------------------------------------------------------------------
def lerp_color(a: RGB, b: RGB, t: float) -> RGB:
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def _spread(index: int, total: int) -> float:
    """Spread customer `index` (0-based) evenly over [0, 1]."""
    if total <= 1:
        return 0.5
    return index / (total - 1)


def customer_start_color(index: int, total: int, theme: Theme) -> RGB:
    return lerp_color(theme.start_shade_lo, theme.start_shade_hi, _spread(index, total))


def customer_end_color(index: int, total: int, theme: Theme) -> RGB:
    return lerp_color(theme.end_shade_lo, theme.end_shade_hi, _spread(index, total))


def delivery_path_color(index: int, theme: Theme) -> RGB:
    """A distinct, saturated hue per customer for its loaded (delivery) leg."""
    hue = (0.06 + 0.3819 * index) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, theme.path_value)
    return (int(r * 255), int(g * 255), int(b * 255))
