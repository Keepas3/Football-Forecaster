"""Where a player plays, in the words the site shows everywhere (the Players page, a team's
squad, the Injuries tab). The four broad ones are football-data.org's, which ESPN's roster labels
follow too, so the pages read alike. Each source spells it its own way; this is the one place they
are turned into these words, and into the attack/defence split the rating model uses.

Some sources can say more than the four broad positions (American Soccer Analysis, for MLS), and
the manual injury list can say anything: those use the finer words below, each of which belongs to
one of the broad groups.
"""

from __future__ import annotations

import re

GOALKEEPER, DEFENCE, MIDFIELD, OFFENCE = "Goalkeeper", "Defence", "Midfield", "Offence"
# The usual goalkeeper-out order, for sorting.
POSITION_ORDER = {GOALKEEPER: 0, DEFENCE: 1, MIDFIELD: 2, OFFENCE: 3}

CENTRE_BACK, LEFT_BACK, RIGHT_BACK, FULL_BACK = "Centre-back", "Left-back", "Right-back", "Full-back"
DEFENSIVE_MID, CENTRAL_MID, ATTACKING_MID = "Defensive mid", "Central mid", "Attacking mid"
WINGER, LEFT_WINGER, RIGHT_WINGER, STRIKER = "Winger", "Left winger", "Right winger", "Striker"

# Every word a position can be shown as, with the broad group it belongs to. The order is the one
# to sort by: each broad word, then the finer ones in its group.
GROUP = {
    GOALKEEPER: GOALKEEPER,
    DEFENCE: DEFENCE,
    CENTRE_BACK: DEFENCE,
    LEFT_BACK: DEFENCE,
    RIGHT_BACK: DEFENCE,
    FULL_BACK: DEFENCE,
    MIDFIELD: MIDFIELD,
    DEFENSIVE_MID: MIDFIELD,
    CENTRAL_MID: MIDFIELD,
    ATTACKING_MID: MIDFIELD,
    OFFENCE: OFFENCE,
    WINGER: OFFENCE,
    LEFT_WINGER: OFFENCE,
    RIGHT_WINGER: OFFENCE,
    STRIKER: OFFENCE,
}
ROLE_ORDER = {word: index for index, word in enumerate(GROUP)}

# The two fantasy feeds (Fantasy Premier League's element_type, Biwenger's position) number
# positions the same way.
FANTASY_POSITIONS = {1: GOALKEEPER, 2: DEFENCE, 3: MIDFIELD, 4: OFFENCE}

# American Soccer Analysis's general_position. It has no left and right: a full-back is one or the other.
ASA_POSITIONS = {
    "GK": GOALKEEPER,
    "CB": CENTRE_BACK,
    "FB": FULL_BACK,
    "DM": DEFENSIVE_MID,
    "CM": CENTRAL_MID,
    "AM": ATTACKING_MID,
    "W": WINGER,
    "ST": STRIKER,
}

# What someone might type for each word, besides the word itself (see normalize_role).
_SPELLINGS = {
    GOALKEEPER: ("goalie", "keeper", "gk"),
    DEFENCE: ("defense", "defender", "def"),
    CENTRE_BACK: ("center back", "centreback", "centerback", "central defender", "cb"),
    LEFT_BACK: ("leftback", "lb"),
    RIGHT_BACK: ("rightback", "rb"),
    FULL_BACK: ("fullback", "wing back", "wingback", "fb"),
    MIDFIELD: ("midfielder", "mid", "mf"),
    DEFENSIVE_MID: ("defensive midfield", "defensive midfielder", "holding midfielder", "cdm", "dm"),
    CENTRAL_MID: ("central midfield", "central midfielder", "cm"),
    ATTACKING_MID: ("attacking midfield", "attacking midfielder", "cam", "am"),
    OFFENCE: ("offense", "forward", "attacker", "fw"),
    WINGER: ("wide forward",),
    LEFT_WINGER: ("leftwinger", "left wing", "lw"),
    RIGHT_WINGER: ("rightwinger", "right wing", "rw"),
    STRIKER: ("centre forward", "center forward", "centreforward", "centerforward", "cf", "st"),
}


def _fold(text: str) -> str:
    """"Left-Back" and "left back" are the same: lower case, words split on anything but letters."""
    return " ".join(re.sub(r"[^a-z]+", " ", text.casefold()).split())


_BY_SPELLING = {_fold(spelling): word for word, spellings in _SPELLINGS.items() for spelling in spellings}
_BY_SPELLING.update({_fold(word): word for word in GROUP})


def normalize_role(text: str | None) -> str:
    """One of GROUP's words for whatever was typed ("cdm", "left back", "Striker"), or "" if it is
    not a position this knows."""
    return _BY_SPELLING.get(_fold(text or ""), "")


def bucket(position: str | None) -> str:
    """"attack" | "defense": the side of the rating an absence moves. Goalkeepers and defenders
    count as defence, everyone else (and anyone unknown) as attack."""
    return "defense" if GROUP.get(position) in (GOALKEEPER, DEFENCE) else "attack"
