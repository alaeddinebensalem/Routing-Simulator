"""A tiny immediate-ish widget toolkit for the side panel.

Nothing here knows about the grid: widgets take callbacks and callables, so the
app can hand them live text and read back changes.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import pygame

import theme as T
from theme import Theme

TextLike = str | Callable[[], str]


def resolve(text: TextLike) -> str:
    return text() if callable(text) else str(text)


class Widget:
    """Base widget: knows its on-screen rect, handles events, draws itself."""

    def __init__(self, height: int) -> None:
        self.height = height
        self.rect = pygame.Rect(0, 0, 0, height)
        self.visible = True

    def layout(self, x: int, y: int, width: int) -> None:
        self.rect = pygame.Rect(x, y, width, self.height)

    def handle_event(self, event: pygame.event.Event) -> bool:
        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        raise NotImplementedError


# ----------------------------------------------------------------- basics
class Label(Widget):
    STYLES = {
        "title": (T.FONT_SIZE_TITLE, True),
        "section": (T.FONT_SIZE_SECTION, True),
        "body": (T.FONT_SIZE_BODY, False),
        "muted": (T.FONT_SIZE_SMALL, False),
    }

    def __init__(self, text: TextLike, style: str = "body") -> None:
        self.style = style if style in self.STYLES else "body"
        size, self.bold = self.STYLES[self.style]
        self.size = size
        super().__init__(size + 6)
        self.text = text

    def _color(self, theme: Theme):
        if self.style == "title":
            return theme.title
        if self.style == "section":
            return theme.accent
        if self.style == "muted":
            return theme.text_muted
        return theme.text

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        font = T.get_font(self.size, self.bold)
        image = font.render(resolve(self.text), True, self._color(theme))
        surface.blit(image, (self.rect.x, self.rect.y + (self.rect.h - image.get_height()) // 2))


class Separator(Widget):
    def __init__(self) -> None:
        super().__init__(T.SEPARATOR_HEIGHT)

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        y = self.rect.centery
        pygame.draw.line(surface, theme.panel_border, (self.rect.x, y), (self.rect.right, y), 1)


class Button(Widget):
    def __init__(self, text: TextLike, on_click: Callable[[], None]) -> None:
        super().__init__(T.BUTTON_HEIGHT)
        self.text = text
        self.on_click = on_click
        self.hovered = False
        self.pressed = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.rect.collidepoint(event.pos):
                self.on_click()
                return True
        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        if self.pressed:
            color = theme.button_press
        elif self.hovered:
            color = theme.button_hover
        else:
            color = theme.button_bg
        pygame.draw.rect(surface, color, self.rect, border_radius=T.CORNER_RADIUS)
        font = T.get_font(T.FONT_SIZE_BODY)
        image = font.render(resolve(self.text), True, theme.button_text)
        surface.blit(image, image.get_rect(center=self.rect.center))


class Row(Widget):
    """Lays a few widgets out side by side in one row."""

    def __init__(self, children: Sequence[Widget], gap: int = 8) -> None:
        super().__init__(max(child.height for child in children))
        self.children = list(children)
        self.gap = gap

    def layout(self, x: int, y: int, width: int) -> None:
        super().layout(x, y, width)
        total_gap = self.gap * (len(self.children) - 1)
        each = (width - total_gap) // len(self.children)
        cursor = x
        for child in self.children:
            child.layout(cursor, y, each)
            cursor += each + self.gap

    def handle_event(self, event: pygame.event.Event) -> bool:
        handled = False
        for child in self.children:
            handled = child.handle_event(event) or handled
        return handled

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        for child in self.children:
            child.draw(surface, theme)


class Toggle(Widget):
    def __init__(self, label: TextLike, value: bool, on_change: Callable[[bool], None]) -> None:
        super().__init__(T.TOGGLE_HEIGHT)
        self.label = label
        self.value = value
        self.on_change = on_change
        self.hovered = False

    @property
    def _switch_rect(self) -> pygame.Rect:
        w, h = 44, 22
        return pygame.Rect(self.rect.right - w, self.rect.centery - h // 2, w, h)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value
                self.on_change(self.value)
                return True
        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        font = T.get_font(T.FONT_SIZE_BODY)
        image = font.render(resolve(self.label), True, theme.text)
        surface.blit(image, (self.rect.x, self.rect.centery - image.get_height() // 2))

        track = self._switch_rect
        color = theme.accent if self.value else theme.toggle_off
        pygame.draw.rect(surface, color, track, border_radius=track.h // 2)
        knob_x = track.right - track.h // 2 if self.value else track.x + track.h // 2
        knob_color = theme.panel_bg if self.value else theme.text_muted
        pygame.draw.circle(surface, knob_color, (knob_x, track.centery), track.h // 2 - 4)


class Slider(Widget):
    def __init__(
        self,
        label: TextLike,
        minimum: float,
        maximum: float,
        value: float,
        on_change: Callable[[float], None],
        step: float | None = None,
        fmt: Callable[[float], str] | None = None,
    ) -> None:
        super().__init__(T.SLIDER_HEIGHT)
        self.label = label
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = step
        self.on_change = on_change
        self.fmt = fmt or (lambda v: f"{v:.2f}")
        self.value = self._quantize(value)
        self.dragging = False

    def _quantize(self, value: float) -> float:
        value = max(self.minimum, min(self.maximum, value))
        if self.step:
            steps = round((value - self.minimum) / self.step)
            value = self.minimum + steps * self.step
            value = max(self.minimum, min(self.maximum, value))
        return value

    def set_value(self, value: float, notify: bool = True) -> None:
        new_value = self._quantize(value)
        if new_value != self.value:
            self.value = new_value
            if notify:
                self.on_change(self.value)

    def set_range(self, minimum: float, maximum: float) -> None:
        self.minimum, self.maximum = float(minimum), float(maximum)
        self.set_value(self.value, notify=False)

    @property
    def _track_rect(self) -> pygame.Rect:
        return pygame.Rect(self.rect.x, self.rect.bottom - 14, self.rect.w, 6)

    def _value_from_pos(self, x: int) -> float:
        track = self._track_rect
        if track.w <= 0:
            return self.minimum
        t = (x - track.x) / track.w
        return self.minimum + (self.maximum - self.minimum) * max(0.0, min(1.0, t))

    def handle_event(self, event: pygame.event.Event) -> bool:
        hit = pygame.Rect(self.rect.x, self.rect.bottom - 24, self.rect.w, 24)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if hit.collidepoint(event.pos):
                self.dragging = True
                self.set_value(self._value_from_pos(event.pos[0]))
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_value(self._value_from_pos(event.pos[0]))
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            return True
        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        font = T.get_font(T.FONT_SIZE_BODY)
        label = font.render(resolve(self.label), True, theme.text)
        surface.blit(label, (self.rect.x, self.rect.y))
        value_text = font.render(self.fmt(self.value), True, theme.text_muted)
        surface.blit(value_text, (self.rect.right - value_text.get_width(), self.rect.y))

        track = self._track_rect
        pygame.draw.rect(surface, theme.slider_track, track, border_radius=3)
        span = self.maximum - self.minimum
        t = 0.0 if span <= 0 else (self.value - self.minimum) / span
        filled = pygame.Rect(track.x, track.y, int(track.w * t), track.h)
        pygame.draw.rect(surface, theme.accent, filled, border_radius=3)
        knob_x = track.x + int(track.w * t)
        pygame.draw.circle(surface, theme.slider_knob, (knob_x, track.centery), 8)


# ------------------------------------------------------------------ panel
class Panel:
    """Auto-stacks widgets vertically; scrolls when taller than its rect."""

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.widgets: list[Widget] = []
        self.scroll = 0

    # -- construction helpers
    def add(self, widget: Widget) -> Widget:
        self.widgets.append(widget)
        self.layout()
        return widget

    def add_title(self, text: TextLike) -> Label:
        return self.add(Label(text, "title"))  # type: ignore[return-value]

    def add_section(self, text: TextLike) -> Label:
        self.add(Separator())
        return self.add(Label(text, "section"))  # type: ignore[return-value]

    def add_label(self, text: TextLike, style: str = "body") -> Label:
        return self.add(Label(text, style))  # type: ignore[return-value]

    def add_separator(self) -> Separator:
        return self.add(Separator())  # type: ignore[return-value]

    def add_button(self, text: TextLike, on_click: Callable[[], None]) -> Button:
        return self.add(Button(text, on_click))  # type: ignore[return-value]

    def add_row(self, children: Iterable[Widget]) -> Row:
        return self.add(Row(list(children)))  # type: ignore[return-value]

    def add_toggle(self, label: TextLike, value: bool, on_change) -> Toggle:
        return self.add(Toggle(label, value, on_change))  # type: ignore[return-value]

    def add_slider(self, *args, **kwargs) -> Slider:
        return self.add(Slider(*args, **kwargs))  # type: ignore[return-value]

    # -- geometry
    @property
    def content_height(self) -> int:
        total = 2 * T.PADDING
        for widget in self.widgets:
            total += widget.height + T.WIDGET_GAP
        return total - T.WIDGET_GAP if self.widgets else total

    @property
    def max_scroll(self) -> int:
        return max(0, self.content_height - self.rect.h)

    def layout(self) -> None:
        x = self.rect.x + T.PADDING
        width = self.rect.w - 2 * T.PADDING
        y = self.rect.y + T.PADDING - self.scroll
        for widget in self.widgets:
            widget.layout(x, y, width)
            y += widget.height + T.WIDGET_GAP

    def set_rect(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.scroll = min(self.scroll, self.max_scroll)
        self.layout()

    # -- runtime
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEWHEEL and self.max_scroll > 0:
            mouse = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse):
                self.scroll = max(0, min(self.max_scroll, self.scroll - event.y * 36))
                self.layout()
                return True
        # Never short-circuit: every widget must see mouse motion.
        handled = False
        for widget in self.widgets:
            handled = widget.handle_event(event) or handled
        return handled

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        pygame.draw.rect(surface, theme.panel_bg, self.rect)
        pygame.draw.line(
            surface, theme.panel_border,
            (self.rect.x, self.rect.y), (self.rect.x, self.rect.bottom), 1,
        )
        previous_clip = surface.get_clip()
        surface.set_clip(self.rect)
        for widget in self.widgets:
            if widget.rect.bottom >= self.rect.y and widget.rect.y <= self.rect.bottom:
                widget.draw(surface, theme)
        surface.set_clip(previous_clip)

        if self.max_scroll > 0:
            track_h = self.rect.h
            bar_h = max(30, int(track_h * self.rect.h / self.content_height))
            t = self.scroll / self.max_scroll
            bar_y = self.rect.y + int((track_h - bar_h) * t)
            bar = pygame.Rect(self.rect.right - 5, bar_y, 3, bar_h)
            pygame.draw.rect(surface, theme.panel_border, bar, border_radius=2)
