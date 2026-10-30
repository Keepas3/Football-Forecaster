"""Matching a feed's club name to this app's own ("Atlético" -> "Atletico Madrid").

Shared by the injury feeds that name clubs their own way (ingest/biwenger_client.py,
ingest/starting11_client.py). Wrong is worse than missing -- an injury list
attached to the wrong club skews its predictions -- so a name is only matched
when it is clear.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz, process

FUZZY_THRESHOLD = 95


def strip_accents(name: str) -> str:
    """"Pongračić" -> "Pongracic": a feed's spelling against a stats site that drops the marks."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


def fold_name(name: str) -> str:
    """Lowercase, accents removed, punctuation turned into spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", strip_accents(name).lower()).split())


def resolve_club_name(
    feed_name: str, our_team_names: list[str], overrides: dict[str, str] | None = None
) -> str | None:
    """Our name for a feed's club, or None if nothing matches clearly.
    `overrides` maps a folded feed name to our name, for the ones words alone
    can't match (a translation, an abbreviation, a different spelling). Otherwise
    the words of one name must be all of the other's ("Lens" in "Racing Club de
    Lens"), and if several of ours fit equally well ("Manchester" alone) none is
    taken."""
    wanted = fold_name((overrides or {}).get(fold_name(feed_name), feed_name))
    by_folded = {fold_name(name): name for name in our_team_names}
    if wanted in by_folded:  # an exact match must never lose to a lookalike
        return by_folded[wanted]
    scored = process.extract(wanted, list(by_folded), scorer=fuzz.token_set_ratio, limit=2)
    if not scored or scored[0][1] < FUZZY_THRESHOLD:
        return None
    if len(scored) > 1 and scored[1][1] >= scored[0][1]:
        return None
    return by_folded[scored[0][0]]
