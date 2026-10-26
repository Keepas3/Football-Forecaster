"""A starting XI drawn on a pitch, from ESPN's formation and position labels.

ESPN gives each starter a position code ("G", "CD-L", "CM-R", "AM", "CF-R", "F"
...) and the team a formation string ("3-4-2-1"). The codes say how far up the
pitch a player plays and on which side; the formation says how many players
are in each line. Players are sorted by depth then side and cut into lines by
the formation's numbers, so each line lines up left to right. The result is an
inline SVG, with every piece of text HTML-escaped.
"""

from __future__ import annotations

import html
import re

from soccer_predictor.ingest.espn_client import LineupPlayer, TeamMatchInfo

# How far from its own goal a position plays (a higher number is further up
# the pitch). Plain "CF" and "F" lead the attack; a "CF" with a side suffix
# ("CF-L") is one of the players just behind the striker.
_DEPTH = {
    "G": 0,
    "SW": 1, "CB": 1, "CD": 1, "LCB": 1, "RCB": 1, "D": 1, "LB": 1, "RB": 1,
    "LWB": 1.5, "RWB": 1.5, "WB": 1.5,
    "DM": 2, "CDM": 2,
    "M": 3, "CM": 3, "LM": 3, "RM": 3,
    "AM": 4, "CAM": 4, "LAM": 4, "RAM": 4, "SS": 4,
    "F": 5, "CF": 5, "ST": 5, "LF": 5, "RF": 5, "LW": 5, "RW": 5,
}
_WIDE = {"LB", "RB", "LWB", "RWB", "LM", "RM", "LW", "RW", "LAM", "RAM", "LF", "RF"}

VIEW_WIDTH = 400
VIEW_HEIGHT = 560
GOALKEEPER_Y = 505
TOP_LINE_Y = 85
LOWEST_OUTFIELD_Y = 405
RADIUS = 21


def _split(code: str) -> tuple[str, str]:
    base, _, suffix = code.upper().partition("-")
    return base, suffix


def depth(code: str, is_second_striker: bool = False) -> float:
    base, suffix = _split(code)
    if base == "CF" and suffix in ("L", "R"):
        return 4.5
    return _DEPTH.get(base, 3)


def side(code: str) -> int:
    """Left to right: -2 wide left, -1 left of centre, 0 centre, 1, 2."""
    base, suffix = _split(code)
    if base.startswith("L") and base in _WIDE:
        return -2
    if base.startswith("R") and base in _WIDE:
        return 2
    return {"L": -1, "R": 1}.get(suffix, 0)


def formation_lines(formation: str | None) -> list[int] | None:
    """"4-2-3-1" -> [4, 2, 3, 1]; None when it isn't of that shape."""
    if not formation or not re.fullmatch(r"\d(-\d){1,5}", formation):
        return None
    return [int(n) for n in formation.split("-")]


def starting_lines(team: TeamMatchInfo) -> list[list[LineupPlayer]]:
    """The starters as lines from the goalkeeper up, each ordered left to right
    (the goalkeeper is the first line). Empty when there is no lineup."""
    starters = [p for p in team.players if p.starter]
    if not starters:
        return []
    keepers = [p for p in starters if _split(p.position)[0] == "G"]
    keeper = keepers[0] if keepers else min(starters, key=lambda p: p.formation_place or 99)
    outfield = [p for p in starters if p is not keeper]
    outfield.sort(key=lambda p: (depth(p.position), side(p.position), p.formation_place or 99))

    sizes = formation_lines(team.formation)
    if sizes and sum(sizes) == len(outfield):
        lines, start = [], 0
        for size in sizes:
            lines.append(outfield[start : start + size])
            start += size
    else:  # no usable formation: one line per distinct depth
        lines = []
        for player in outfield:
            if lines and depth(lines[-1][0].position) == depth(player.position):
                lines[-1].append(player)
            else:
                lines.append([player])
    ordered = [sorted(line, key=lambda p: (side(p.position), p.formation_place or 99)) for line in lines]
    return [[keeper], *ordered]


def _label(name: str) -> str:
    words = name.split()
    return (words[-1] if words else name).upper()[:12]


def short_minute(label: str) -> str:
    """ESPN's "90'+3'" as "90+3'"."""
    return label.replace("'+", "+")


MINUTE_CHAR_WIDTH = 6.9  # about how wide one character of the minute text is
MINUTE_GAP = RADIUS + 7  # from the player's centre to where the minutes start


def _badges(player: LineupPlayer, x: float) -> str:
    """Tiny markers beside a player's circle (centred at `x` across the pitch): goals,
    assists, cards, and coming off, each with the minute it happened when ESPN gave one.
    The subbed-off arrow is bigger and a strong red with a white outline so it stays
    readable against the grass. Normally on the right; on the left for a player so near the
    right edge that the minutes would run off the pitch."""
    marks = []  # (icon, colour, font size, extra attributes, minute labels)
    minutes = player.event_minutes
    if player.stats.get("totalGoals"):
        goals, times = player.stats["totalGoals"], minutes.get("goal", [])
        marks.append(("⚽" if goals == 1 or len(times) == goals else f"⚽{goals}", "#ffffff", 12, "", times))
    if player.stats.get("goalAssists"):
        assists, times = player.stats["goalAssists"], minutes.get("assist", [])
        marks.append(("👟" if assists == 1 or len(times) == assists else f"👟{assists}", "#ffffff", 12, "", times))
    if player.stats.get("redCards"):
        marks.append(("■", "#e5342f", 12, "", minutes.get("red", [])))
    elif player.stats.get("yellowCards"):
        marks.append(("■", "#f2c200", 12, "", minutes.get("yellow", [])))
    if player.subbed_out:
        arrow = ' font-weight="900" stroke="#ffffff" stroke-width="3" paint-order="stroke"'
        marks.append(("↓", "#c62828", 20, arrow, [player.sub_minute] if player.sub_minute else []))
    labels = [" ".join(short_minute(time) for time in times) for *_, times in marks]
    longest = max((len(label) for label in labels), default=0)
    mirror = x + MINUTE_GAP + longest * MINUTE_CHAR_WIDTH > VIEW_WIDTH - 6
    sign = -1 if mirror else 1
    parts, y = [], -RADIUS + 8
    for (icon, colour, size, extra, times), label in zip(marks, labels):
        parts.append(
            f'<text x="{sign * (RADIUS - 2)}" y="{y}" font-size="{size}" fill="{colour}" text-anchor="middle"{extra}>'
            f"{html.escape(icon)}</text>"
        )
        if label:
            parts.append(
                f'<text x="{sign * MINUTE_GAP}" y="{y - 1 - (size - 12) * 0.4:.1f}" font-size="12" font-weight="700" '
                f'fill="#ffffff" stroke="#0b1d3a" stroke-width="3" stroke-linejoin="round" paint-order="stroke" '
                f'text-anchor="{"end" if mirror else "start"}" font-family="sans-serif">{html.escape(label)}</text>'
            )
        y += size + 3
    return "".join(parts)


def pitch_svg(team: TeamMatchInfo) -> str:
    """The team's starting XI as a vertical pitch, goalkeeper at the bottom. ""
    when there is no lineup (the caller shows a note instead)."""
    lines = starting_lines(team)
    if not lines:
        return ""
    parts = [
        f'<svg viewBox="0 0 {VIEW_WIDTH} {VIEW_HEIGHT}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:420px;height:auto;display:block;margin:0 auto" role="img" '
        f'aria-label="{html.escape(team.name)} starting lineup, {html.escape(team.formation or "")}">',
        f'<rect width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}" rx="10" fill="#2e7d32"/>',
    ]
    for i in range(8):  # mowing stripes
        if i % 2 == 0:
            parts.append(f'<rect y="{i * 70}" width="{VIEW_WIDTH}" height="70" fill="#378a3b"/>')
    line = 'fill="none" stroke="#ffffffcc" stroke-width="2"'
    parts += [
        f'<rect x="12" y="12" width="{VIEW_WIDTH - 24}" height="{VIEW_HEIGHT - 24}" {line}/>',
        f'<line x1="12" y1="{VIEW_HEIGHT // 2}" x2="{VIEW_WIDTH - 12}" y2="{VIEW_HEIGHT // 2}" stroke="#ffffffcc" stroke-width="2"/>',
        f'<circle cx="{VIEW_WIDTH // 2}" cy="{VIEW_HEIGHT // 2}" r="48" {line}/>',
        f'<rect x="110" y="12" width="180" height="70" {line}/>',
        f'<rect x="150" y="12" width="100" height="26" {line}/>',
        f'<rect x="110" y="{VIEW_HEIGHT - 82}" width="180" height="70" {line}/>',
        f'<rect x="150" y="{VIEW_HEIGHT - 38}" width="100" height="26" {line}/>',
    ]

    outfield_lines = len(lines) - 1
    badge_layers = []  # drawn after every player, so a neighbour's circle never covers a minute
    for row, players in enumerate(lines):
        if row == 0:
            y = GOALKEEPER_Y
        elif outfield_lines == 1:
            y = (LOWEST_OUTFIELD_Y + TOP_LINE_Y) / 2
        else:
            y = LOWEST_OUTFIELD_Y - (row - 1) * (LOWEST_OUTFIELD_Y - TOP_LINE_Y) / (outfield_lines - 1)
        for index, player in enumerate(players):
            x = VIEW_WIDTH * (index + 1) / (len(players) + 1)
            number = html.escape(re.sub(r"\D", "", player.jersey)[:2])
            parts.append(
                f'<g transform="translate({x:.1f} {y:.1f})">'
                f'<circle r="{RADIUS}" fill="#ffffff" stroke="#0b1d3a" stroke-width="1.5"/>'
                f'<text y="7" font-size="20" font-weight="700" fill="#0b1d3a" text-anchor="middle" '
                f'font-family="sans-serif">{number}</text>'
                f'<rect x="-40" y="{RADIUS + 4}" width="80" height="18" rx="9" fill="#e9ecef" opacity="0.92"/>'
                f'<text y="{RADIUS + 17}" font-size="11" font-weight="700" fill="#2b2f36" text-anchor="middle" '
                f'font-family="sans-serif">{html.escape(_label(player.name))}</text></g>'
            )
            badges = _badges(player, x)
            if badges:
                badge_layers.append(f'<g transform="translate({x:.1f} {y:.1f})">{badges}</g>')
    parts += badge_layers
    parts.append("</svg>")
    return "".join(parts)
