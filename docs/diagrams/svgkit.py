"""A tiny SVG toolkit for the repo's diagrams: dot-grid paper, role colours, cards, pills and arrows."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape

SANS = (
    "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
)
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"

PAPER = "#FBFCFE"
GRID = "#D9E1EC"
INK = "#1F2A44"
MUTED = "#5B6780"

# role -> (fill, stroke). Colour always means the same thing in every diagram.
ROLES: dict[str, tuple[str, str]] = {
    "data": ("#E2F3F1", "#2B7A76"),  # text, tokens, datasets on disk
    "model": ("#E8EDFC", "#3352C2"),  # a network or a computation
    "loss": ("#FBE6EE", "#B0215A"),  # an objective being minimised
    "reward": ("#FBF2D6", "#A97A00"),  # rewards, scores, advantages
    "frozen": ("#F0F2F5", "#6B7280"),  # a copy that is not trained
    "result": ("#E8F3DF", "#4A7A28"),  # what comes out: a checkpoint, a number, text
    "plain": ("#FFFFFF", "#8A96AB"),  # a small operation
}


def text_width(text: str, size: float, mono: bool = False) -> float:
    """Rough rendered width of a string, generous so system fonts never overflow."""
    return len(text) * size * (0.62 if mono else 0.56)


@dataclass
class Node:
    """A box on the page, positioned by its centre."""

    x: float
    y: float
    w: float
    h: float

    def side(self, name: str) -> tuple[float, float]:
        """The middle of one side: 'l', 'r', 't' or 'b'."""
        return {
            "l": (self.x - self.w / 2, self.y),
            "r": (self.x + self.w / 2, self.y),
            "t": (self.x, self.y - self.h / 2),
            "b": (self.x, self.y + self.h / 2),
        }[name]


@dataclass
class Diagram:
    """One figure: a title, a one-line caption and the drawing underneath."""

    width: int
    height: int
    title: str
    caption: str = ""
    top: int = 92
    parts: list[str] = field(default_factory=list)
    overlay: list[str] = field(default_factory=list)

    # --- shapes -------------------------------------------------------------------------

    def card(
        self,
        x: float,
        y: float,
        text: str,
        role: str = "model",
        sub: str = "",
        w: float | None = None,
        h: float | None = None,
        mono_sub: bool = False,
        size: float = 15,
        dashed: bool | None = None,
        stack: bool = False,
        pill: bool = False,
        bold: bool = True,
    ) -> Node:
        """A labelled box; ``stack`` draws stored data, ``pill`` a small operation."""
        lines = text.split("\n")
        subs = sub.split("\n") if sub else []
        sub_size = size - 3
        need = max(
            [text_width(t, size) for t in lines]
            + [text_width(s, sub_size, mono_sub) for s in subs]
        )
        w = w or max(need + 36, 96)
        h = h or (len(lines) * (size + 5) + len(subs) * (sub_size + 5) + 22)
        y = y + self.top
        fill, stroke = ROLES[role]
        dash = (
            ' stroke-dasharray="6 4"'
            if (role == "frozen" if dashed is None else dashed)
            else ""
        )
        r = h / 2 if pill else 9
        x0, y0 = x - w / 2, y - h / 2
        if stack:
            for k in (2, 1):
                self.parts.append(
                    f'<rect x="{x0 + 5 * k:.1f}" y="{y0 - 5 * k:.1f}" width="{w:.1f}" height="{h:.1f}" '
                    f'rx="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.4" opacity="0.75"/>'
                )
        self.parts.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"{dash}/>'
        )
        total = len(lines) * (size + 5) + len(subs) * (sub_size + 5) - 5
        cy = y - total / 2 + size * 0.8
        weight = 600 if bold else 400
        for t in lines:
            self.parts.append(self._text(x, cy, t, size, INK, weight=weight))
            cy += size + 5
        for s in subs:
            self.parts.append(self._text(x, cy, s, sub_size, MUTED, mono=mono_sub))
            cy += sub_size + 5
        return Node(x, y, w, h)

    def group(
        self, x: float, y: float, w: float, h: float, label: str, role: str = "plain"
    ) -> None:
        """A dashed container with a label in its top-left corner."""
        y = y + self.top
        stroke = ROLES[role][1]
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="16" fill="none" '
            f'stroke="{stroke}" stroke-width="1.3" stroke-dasharray="3 5"/>'
        )
        self.parts.append(
            self._text(x + 16, y + 22, label, 13, stroke, anchor="start", weight=600)
        )

    def label(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 13,
        color: str = MUTED,
        mono: bool = False,
        anchor: str = "middle",
        weight: int = 400,
    ) -> None:
        """Free text on the page."""
        self.overlay.append(
            self._text(
                x,
                y + self.top,
                text,
                size,
                color,
                mono=mono,
                anchor=anchor,
                weight=weight,
            )
        )

    def chip(
        self,
        x: float,
        y: float,
        text: str,
        role: str = "plain",
        w: float | None = None,
        size: float = 13,
        mono: bool = True,
    ) -> Node:
        """A small square-ish tile, for a token, an id or a matrix cell."""
        w = w or text_width(text, size, mono) + 16
        h = size + 14
        fill, stroke = ROLES[role]
        yy = y + self.top
        self.parts.append(
            f'<rect x="{x - w / 2:.1f}" y="{yy - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
        )
        self.parts.append(self._text(x, yy + size * 0.36, text, size, INK, mono=mono))
        return Node(x, yy, w, h)

    def badge(self, node: Node, number: int) -> None:
        """A step number on the top-left corner of a card (only for real sequences)."""
        cx, cy = node.x - node.w / 2, node.y - node.h / 2
        self.overlay.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="{INK}"/>'
        )
        self.overlay.append(
            self._text(cx, cy + 4.5, str(number), 12, "#FFFFFF", weight=700)
        )

    # --- connections --------------------------------------------------------------------

    def arrow(
        self,
        a: Node | tuple[float, float],
        b: Node | tuple[float, float],
        label: str = "",
        out: str | None = None,
        into: str | None = None,
        via: list[tuple[float, float]] | None = None,
        dashed: bool = False,
        color: str = INK,
        label_at: float = 0.5,
        label_dy: float = -8,
        head: bool = True,
        mono: bool = True,
        straight: bool = False,
    ) -> None:
        """An orthogonal arrow from ``a`` to ``b`` (nodes or page points), with an optional shape label."""
        first = via[0] if via else b
        last = via[-1] if via else a
        p0 = self._anchor(a, out, first, self.top)
        p1 = self._anchor(b, into, last, self.top)
        pts = [p0] + [(vx, vy + self.top) for vx, vy in (via or [])] + [p1]
        if not via and not straight and p0[0] != p1[0] and p0[1] != p1[1]:
            horizontal_first = (
                out in ("l", "r") if out else abs(p1[0] - p0[0]) >= abs(p1[1] - p0[1])
            )
            if horizontal_first and into in ("t", "b"):
                pts = [p0, (p1[0], p0[1]), p1]
            elif not horizontal_first and into in ("l", "r"):
                pts = [p0, (p0[0], p1[1]), p1]
            elif horizontal_first:
                mx = (p0[0] + p1[0]) / 2
                pts = [p0, (mx, p0[1]), (mx, p1[1]), p1]
            else:
                my = (p0[1] + p1[1]) / 2
                pts = [p0, (p0[0], my), (p1[0], my), p1]
        d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        marker = f' marker-end="url(#{self._marker(color)})"' if head else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.7" '
            f'stroke-linejoin="round"{dash}{marker}/>'
        )
        if label:
            lx, ly = self._along(pts, label_at)
            self.overlay.append(
                self._halo_text(lx, ly + label_dy, label, 12, mono=mono)
            )

    def plus(self, x: float, y: float, r: float = 13) -> Node:
        """A circled plus, for a residual add."""
        yy = y + self.top
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="{r}" fill="#FFFFFF" stroke="{INK}" stroke-width="1.7"/>'
        )
        self.parts.append(
            f'<path d="M {x - 6} {yy} L {x + 6} {yy} M {x} {yy - 6} L {x} {yy + 6}" '
            f'stroke="{INK}" stroke-width="1.8"/>'
        )
        return Node(x, yy, 2 * r, 2 * r)

    # --- output -------------------------------------------------------------------------

    def svg(self) -> str:
        """The finished SVG document."""
        head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{escape(self.title)}">',
            f"<title>{escape(self.title)}</title>",
            "<defs>",
            f'<pattern id="dots" width="18" height="18" patternUnits="userSpaceOnUse">'
            f'<circle cx="1.5" cy="1.5" r="1.1" fill="{GRID}"/></pattern>',
            *self._markers.values(),
            "</defs>",
            f'<rect width="100%" height="100%" fill="{PAPER}"/>',
            '<rect width="100%" height="100%" fill="url(#dots)"/>',
            self._text(32, 42, self.title, 21, INK, anchor="start", weight=700),
        ]
        if self.caption:
            head.append(self._text(32, 66, self.caption, 14, MUTED, anchor="start"))
        return "\n".join(head + self.parts + self.overlay + ["</svg>"]) + "\n"

    # --- internals ----------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Start with no arrowhead markers."""
        self._markers: dict[str, str] = {}

    def _marker(self, color: str) -> str:
        name = "head" + color.strip("#")
        if name not in self._markers:
            self._markers[name] = (
                f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
                f'markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
                f'fill="{color}"/></marker>'
            )
        return name

    @staticmethod
    def _anchor(
        n: Node | tuple[float, float],
        side: str | None,
        other: Node | tuple[float, float],
        top: float,
    ) -> tuple[float, float]:
        if isinstance(n, tuple):
            return n[0], n[1] + top
        if side is None:
            ox, oy = (
                (other.x, other.y)
                if isinstance(other, Node)
                else (other[0], other[1] + top)
            )
            dx, dy = ox - n.x, oy - n.y
            side = (
                ("r" if dx > 0 else "l")
                if abs(dx) * n.h > abs(dy) * n.w
                else ("b" if dy > 0 else "t")
            )
        return n.side(side)

    @staticmethod
    def _along(pts: list[tuple[float, float]], t: float) -> tuple[float, float]:
        seg = [
            ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            for a, b in zip(pts, pts[1:])
        ]
        goal, run = sum(seg) * t, 0.0
        for (a, b), s in zip(zip(pts, pts[1:]), seg):
            if run + s >= goal and s:
                f = (goal - run) / s
                return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            run += s
        return pts[-1]

    @staticmethod
    def _text(
        x: float,
        y: float,
        text: str,
        size: float,
        color: str,
        mono: bool = False,
        anchor: str = "middle",
        weight: int = 400,
    ) -> str:
        family = MONO if mono else SANS
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    def _halo_text(
        self, x: float, y: float, text: str, size: float, mono: bool = True
    ) -> str:
        w = text_width(text, size, mono) + 10
        return (
            f'<rect x="{x - w / 2:.1f}" y="{y - size + 1:.1f}" width="{w:.1f}" height="{size + 7}" rx="3" '
            f'fill="{PAPER}" opacity="0.95"/>'
            + self._text(x, y + 2, text, size, MUTED, mono=mono)
        )
