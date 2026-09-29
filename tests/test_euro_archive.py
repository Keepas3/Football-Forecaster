from __future__ import annotations

import datetime as dt

from soccer_predictor.ingest import euro_archive

# Real excerpt from the 1996 file (openfootball/euro, 1996--england/euro.txt,
# fetched live 2026-09) -- covers the group-stage "date @ city team score
# team" single-line layout and the knockout "date/venue on one line, score
# with (aet, N-N pen) on the next" layout, plus the Final's real
# goal-scorer/lineup block (the only match in the whole file that had one --
# confirmed live, matches this module's documented sparse-coverage finding).
EURO_1996_EXCERPT = """
▪ Group A
June 8   @ London            England          1-1 Switzerland
June 10  @ Birmingham        Netherlands      0-0 Scotland
June 18  @ Birmingham        Scotland         1-0 Switzerland
June 18  @ London            Netherlands      1-4 England

▪ Quarter-finals
June 22 15:00 @ London
  England          0-0 Spain           (aet, 4-2 pen)
June 22 18:30 @ Liverpool
  France           0-0 Netherlands     (aet, 5-4 pen)

▪ Final
June 30 19:00 @ London
  Germany          2-1 Czech Republic   (aet/gg)
    (Bierhoff 73',95'; Berger 59'(pen))
    Germany: Köpke, Sammer, Babbel, Helmer, Strunz, Hässler, Eilts (Bode 46'),
         Scholl (Bierhoff 69'), Ziege, Klinsmann, Kuntz
    Czech Republic: Kouba, Kadlec, Hornak, Suchoparek, Poborsky (Smicer 88'),
                Nedved, Rada, Berger, Bejbl, Nemec, Kuka
"""

# Real excerpt from the 1972 file (1972--belgium/euro.txt) -- the
# "date+venue on its own line, then a bare TeamA score TeamB line with no
# date/venue prefix at all" layout, plus a real multi-player goal block
# ("G.Müller 24, 71" = ONE player scoring twice; "; Polleunis 83" = a
# different player on the other side).
EURO_1972_EXCERPT = """
▪ Semi-finals
June 14 20:00 @ Antwerpen, Bosuil
West Germany     2-1 Belgium
  (G.Müller 24, 71; Polleunis 83)
  West Germany: Maier, Höttges, Beckenbauer
"""

# Real excerpt from the 1960 file (1960--france/euro.txt) -- the oldest
# layout: "TeamA v TeamB N-N @ Venue" (score AFTER both team names).
EURO_1960_EXCERPT = """
▪ Semi-finals
July 6
  20:00    France  v Yugoslavia   4-5            @ Parc des Princes, Paris
  20:30    Czechoslovakia  v Soviet Union  0-3    @ Stade Vélodrome, Marseille
"""

# Real excerpt from the 1980 file (1980--italy/euro.txt) -- a match followed
# by an un-parenthesized penalty-shootout breakdown, which used to get
# mis-parsed as two extra phantom "matches" ("Penalties:" 0-1 "Causio, 1-1
# Masny,", and "Collovati (saved)," 9-8 "Barmos") since both lines contain
# their own N-N-shaped substring.
EURO_1980_PENALTIES_EXCERPT = """
▪ Third place play-off
June 21 @ Napoli
Czechoslovakia   1-1 Italy    (9-8 pen)        # note - no extra time played
  Penalties: 0-1 Causio, 1-1 Masny,
             2-1 Panenka, 2-2 Antognoni,
                 Collovati (saved), 9-8 Barmos
"""

# Real excerpt from the 1976 file (1976--yugoslavia/euro.txt) -- "a.e.t."
# (no parens) sits between the score and the real away team name.
EURO_1976_AET_EXCERPT = """
▪ Semi-finals
June 16 20:15 @ Zagreb, Maksimir Stadion
Czechoslovakia   3-1 a.e.t.  Netherlands
"""

# Real excerpt from the 2012 file (2012--poland-ukraine/euro.txt) -- a
# kickoff time + timezone annotation separated from the team name by only a
# single space (not the usual 2+-space alignment gap).
EURO_2012_TIMEZONE_PREFIX_EXCERPT = """
▪ Group B
  20:45 (21:45 EEST) Germany        1-0  Portugal    @ Lviv
"""

# Real excerpt from the 2016 file (2016--france/euro.txt) -- the "date on
# its own line, then an indented time+score+venue line" layout, including
# two matches sharing one date.
EURO_2016_EXCERPT = """
▪ Group A
Jun 10
  21:00   France            2-1  Romania           @ Paris (Stade de France)
Jun 19
  21:00   Romania           0-1  Albania           @ Lille
          Switzerland       0-0  France            @ Lyon
"""


def test_parse_matches_1996_group_stage_single_line_layout():
    matches = euro_archive.parse_matches(EURO_1996_EXCERPT, 1996)

    england_switzerland = next(m for m in matches if m["home_team_name"] == "England")
    assert england_switzerland["away_team_name"] == "Switzerland"
    assert england_switzerland["home_goals"] == 1
    assert england_switzerland["away_goals"] == 1
    assert england_switzerland["date"] == dt.date(1996, 6, 8)


def test_parse_matches_1996_knockout_layout_uses_last_seen_date():
    matches = euro_archive.parse_matches(EURO_1996_EXCERPT, 1996)

    quarter_final = next(m for m in matches if m["home_team_name"] == "England" and m["away_team_name"] == "Spain")
    assert quarter_final["home_goals"] == 0
    assert quarter_final["away_goals"] == 0
    assert quarter_final["date"] == dt.date(1996, 6, 22)


def test_parse_matches_1996_final_ignores_aet_gg_annotation():
    matches = euro_archive.parse_matches(EURO_1996_EXCERPT, 1996)

    final = next(m for m in matches if m["home_team_name"] == "Germany")
    assert final["away_team_name"] == "Czech Republic"
    assert final["home_goals"] == 2
    assert final["away_goals"] == 1
    assert final["date"] == dt.date(1996, 6, 30)


def test_parse_matches_1996_finds_all_seven_matches():
    matches = euro_archive.parse_matches(EURO_1996_EXCERPT, 1996)
    assert len(matches) == 7


def test_parse_matches_1972_date_on_separate_line_from_bare_match():
    matches = euro_archive.parse_matches(EURO_1972_EXCERPT, 1972)

    assert len(matches) == 1
    match = matches[0]
    assert match["home_team_name"] == "West Germany"
    assert match["away_team_name"] == "Belgium"
    assert match["home_goals"] == 2
    assert match["away_goals"] == 1
    assert match["date"] == dt.date(1972, 6, 14)


def test_parse_matches_1960_v_format_score_after_team_names():
    matches = euro_archive.parse_matches(EURO_1960_EXCERPT, 1960)

    assert len(matches) == 2
    france = next(m for m in matches if m["home_team_name"] == "France")
    assert france["away_team_name"] == "Yugoslavia"
    assert france["home_goals"] == 4
    assert france["away_goals"] == 5
    assert france["date"] == dt.date(1960, 7, 6)

    czechoslovakia = next(m for m in matches if m["home_team_name"] == "Czechoslovakia")
    assert czechoslovakia["away_team_name"] == "Soviet Union"
    assert czechoslovakia["home_goals"] == 0
    assert czechoslovakia["away_goals"] == 3


def test_parse_matches_2016_indented_time_score_venue_layout():
    matches = euro_archive.parse_matches(EURO_2016_EXCERPT, 2016)

    assert len(matches) == 3
    france_romania = next(m for m in matches if m["home_team_name"] == "France" and m["away_team_name"] == "Romania")
    assert france_romania["home_goals"] == 2
    assert france_romania["away_goals"] == 1
    assert france_romania["date"] == dt.date(2016, 6, 10)


def test_parse_matches_2016_two_matches_share_one_date_line():
    matches = euro_archive.parse_matches(EURO_2016_EXCERPT, 2016)

    romania_albania = next(m for m in matches if m["home_team_name"] == "Romania" and m["away_team_name"] == "Albania")
    switzerland_france = next(m for m in matches if m["home_team_name"] == "Switzerland")
    assert romania_albania["date"] == dt.date(2016, 6, 19)
    assert switzerland_france["date"] == dt.date(2016, 6, 19)
    assert switzerland_france["away_team_name"] == "France"
    assert switzerland_france["home_goals"] == 0
    assert switzerland_france["away_goals"] == 0


def test_parse_matches_skips_unparenthesized_penalty_breakdown_as_phantom_matches():
    matches = euro_archive.parse_matches(EURO_1980_PENALTIES_EXCERPT, 1980)

    assert len(matches) == 1
    match = matches[0]
    assert match["home_team_name"] == "Czechoslovakia"
    assert match["away_team_name"] == "Italy"
    assert match["home_goals"] == 1
    assert match["away_goals"] == 1


def test_parse_matches_skips_bare_aet_annotation_between_score_and_away_team():
    matches = euro_archive.parse_matches(EURO_1976_AET_EXCERPT, 1976)

    assert len(matches) == 1
    match = matches[0]
    assert match["home_team_name"] == "Czechoslovakia"
    assert match["away_team_name"] == "Netherlands"
    assert match["home_goals"] == 3
    assert match["away_goals"] == 1


def test_parse_matches_strips_single_space_separated_timezone_prefix():
    matches = euro_archive.parse_matches(EURO_2012_TIMEZONE_PREFIX_EXCERPT, 2012)

    assert len(matches) == 1
    match = matches[0]
    assert match["home_team_name"] == "Germany"
    assert match["away_team_name"] == "Portugal"
    assert match["home_goals"] == 1
    assert match["away_goals"] == 0


def test_parse_matches_falls_back_to_tournament_year_when_no_date_seen():
    matches = euro_archive.parse_matches("Sweden           1-1 France", 1992)
    assert matches[0]["date"] == dt.date(1992, 7, 1)


def test_parse_goalscorers_only_extracts_scorers_from_the_final():
    goals = euro_archive.parse_goalscorers(EURO_1996_EXCERPT, 1996)

    # Only the Final has a scorer block in this real excerpt -- every other
    # match (group stage + quarter-finals) correctly produces nothing.
    assert {g["player_name"] for g in goals} == {"Bierhoff", "Berger"}


def test_parse_goalscorers_same_player_multiple_goals_from_comma_list():
    goals = euro_archive.parse_goalscorers(EURO_1996_EXCERPT, 1996)

    bierhoff_goals = [g for g in goals if g["player_name"] == "Bierhoff"]
    assert len(bierhoff_goals) == 2
    assert {g["minute"] for g in bierhoff_goals} == {73, 95}
    assert all(g["team_name"] == "Germany" for g in bierhoff_goals)


def test_parse_goalscorers_flags_penalty():
    goals = euro_archive.parse_goalscorers(EURO_1996_EXCERPT, 1996)

    berger_goal = next(g for g in goals if g["player_name"] == "Berger")
    assert berger_goal["minute"] == 59
    assert berger_goal["penalty"] is True
    assert berger_goal["team_name"] == "Czech Republic"


def test_parse_goalscorers_distinguishes_two_different_players_on_one_side():
    goals = euro_archive.parse_goalscorers(EURO_1972_EXCERPT, 1972)

    home_goals = [g for g in goals if g["team_name"] == "West Germany"]
    away_goals = [g for g in goals if g["team_name"] == "Belgium"]
    assert {g["player_name"] for g in home_goals} == {"G.Müller"}
    assert {g["minute"] for g in home_goals} == {24, 71}
    assert [(g["player_name"], g["minute"]) for g in away_goals] == [("Polleunis", 83)]


def test_parse_goalscorers_returns_empty_for_text_with_no_scorer_blocks():
    assert euro_archive.parse_goalscorers(EURO_1960_EXCERPT, 1960) == []


def test_list_tournament_folders_includes_2020_override():
    folders = euro_archive.list_tournament_folders()
    assert "2021--europe" in folders
    assert "2020--europe" not in folders
