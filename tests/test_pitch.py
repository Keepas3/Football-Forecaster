"""The starting-XI pitch graphic: line-ups cut into lines, escaping, no lineup."""

from __future__ import annotations

from soccer_predictor.dashboard import pitch
from soccer_predictor.ingest.espn_client import LineupPlayer, TeamMatchInfo


def _team(formation, positions, name="Team"):
    players = [
        LineupPlayer(name=f"Player {chr(65 + i)}{i}", jersey=str(i + 1), position=pos, starter=True, formation_place=i + 1)
        for i, pos in enumerate(positions)
    ]
    return TeamMatchInfo(espn_id="1", name=name, home_away="home", formation=formation, players=players)


def _lines(team):
    return [[p.position for p in line] for line in pitch.starting_lines(team)]


def test_a_back_three_with_two_behind_the_striker():
    team = _team("3-4-2-1", ["G", "RM", "LM", "CD-L", "CD", "CD-R", "CM-R", "CM-L", "F", "CF-R", "CF-L"])
    assert _lines(team) == [
        ["G"],
        ["CD-L", "CD", "CD-R"],
        ["LM", "CM-L", "CM-R", "RM"],  # left to right, wide players outside
        ["CF-L", "CF-R"],
        ["F"],
    ]


def test_a_4_2_3_1_puts_the_attacking_midfielders_in_front_of_the_pivot():
    team = _team("4-2-3-1", ["G", "LB", "CD-L", "CD-R", "RB", "CM-L", "CM-R", "AM-L", "AM", "AM-R", "F"])
    assert _lines(team) == [["G"], ["LB", "CD-L", "CD-R", "RB"], ["CM-L", "CM-R"], ["AM-L", "AM", "AM-R"], ["F"]]


def test_without_a_usable_formation_players_are_grouped_by_how_far_up_they_play():
    team = _team(None, ["G", "LB", "CD-L", "CD-R", "RB", "LM", "CM", "RM", "LF", "F", "RF"])
    assert _lines(team) == [["G"], ["LB", "CD-L", "CD-R", "RB"], ["LM", "CM", "RM"], ["LF", "F", "RF"]]
    # A formation that doesn't add up to the ten outfield players is ignored the same way.
    assert _lines(_team("4-4-3", ["G", "LB", "CD-L", "CD-R", "RB", "LM", "CM", "RM", "LF", "F", "RF"])) == _lines(team)


def test_substitutes_are_not_drawn_and_no_lineup_means_no_graphic():
    team = _team("4-4-2", ["G", "LB", "CD-L", "CD-R", "RB", "LM", "CM-L", "CM-R", "RM", "CF-L", "CF-R"])
    team.players.append(LineupPlayer("Bench Guy", "99", "SUB", starter=False))
    assert "BENCH" not in pitch.pitch_svg(team)
    assert pitch.pitch_svg(_team("4-4-2", [])) == ""


def test_the_svg_has_every_shirt_number_surname_and_badges():
    team = _team("4-4-2", ["G", "LB", "CD-L", "CD-R", "RB", "LM", "CM-L", "CM-R", "RM", "CF-L", "CF-R"], name="Spain")
    team.players[10].stats = {"totalGoals": 2, "goalAssists": 1, "yellowCards": 1}
    team.players[9].stats = {"goalAssists": 3}
    team.players[10].subbed_out = True
    svg = pitch.pitch_svg(team)
    assert svg.startswith("<svg") and svg.endswith("</svg>") and "Spain starting lineup, 4-4-2" in svg
    for player in team.players:
        assert f">{player.jersey}</text>" in svg and player.name.split()[-1].upper() in svg
    assert "⚽2" in svg and "👟3" in svg and ">👟<" in svg and "■" in svg and "↓" in svg


def test_hostile_names_and_numbers_cannot_inject_markup():
    team = _team("4-4-2", ["G", "LB", "CD-L", "CD-R", "RB", "LM", "CM-L", "CM-R", "RM", "CF-L", "CF-R"], name='"><script>x</script>')
    team.players[0].name = '<img src=x onerror=alert(1)>'
    team.players[0].jersey = "<b>7</b>"
    svg = pitch.pitch_svg(team)
    assert "<script" not in svg and "<img" not in svg and "<b>" not in svg
    assert "&lt;" in svg


def _squad(**by_index):
    team = _team("4-4-2", ["G", "LB", "CD-L", "CD-R", "RB", "LM", "CM-L", "CM-R", "RM", "CF-L", "CF-R"])
    for index, fields in by_index.items():
        for name, value in fields.items():
            setattr(team.players[int(index[1:])], name, value)
    return team


def test_each_marker_shows_the_minute_it_happened():
    team = _squad(
        p10=dict(stats={"totalGoals": 2, "goalAssists": 1, "yellowCards": 1}, subbed_out=True, sub_minute="78'",
                 event_minutes={"goal": ["23'", "90'+2'"], "assist": ["61'"], "yellow": ["44'"]}),
        p9=dict(stats={"redCards": 1}, event_minutes={"red": ["88'"]}),
    )
    svg = pitch.pitch_svg(team)
    assert ">23&#x27; 90+2&#x27;<" in svg  # both goals, stoppage time as 90+2'
    assert ">61&#x27;<" in svg and ">44&#x27;<" in svg and ">78&#x27;<" in svg and ">88&#x27;<" in svg
    assert "⚽2" not in svg and ">⚽<" in svg  # the minutes say how many, so no count beside the ball


def test_without_minutes_the_markers_are_as_before():
    team = _squad(p10=dict(stats={"totalGoals": 2, "goalAssists": 1}, subbed_out=True))
    svg = pitch.pitch_svg(team)
    assert "⚽2" in svg and ">👟<" in svg and "↓" in svg and "&#x27;" not in svg


def test_minutes_that_would_run_off_the_right_edge_go_on_the_left_of_the_player():
    hat_trick = {"totalGoals": 3}
    times = {"goal": ["2'", "7'", "21'"]}
    right_back = _squad(p4=dict(stats=hat_trick, event_minutes=times))  # the right-most of four across
    keeper = _squad(p0=dict(stats=hat_trick, event_minutes=times))  # in the middle
    assert 'text-anchor="end"' in pitch.pitch_svg(right_back)
    assert 'text-anchor="end"' not in pitch.pitch_svg(keeper)
    # A short minute still fits on the right at the same spot.
    short = _squad(p4=dict(stats={"totalGoals": 1}, event_minutes={"goal": ["7'"]}))
    assert 'text-anchor="end"' not in pitch.pitch_svg(short)


def test_markers_are_drawn_after_every_player_so_a_neighbour_never_covers_a_minute():
    team = _squad(p5=dict(stats={"totalGoals": 1}, event_minutes={"goal": ["12'"]}))
    svg = pitch.pitch_svg(team)
    last_player = svg.rindex('fill="#2b2f36"')  # the last surname label
    assert svg.index("12&#x27;") > last_player


def test_minutes_are_escaped():
    team = _squad(p10=dict(stats={"totalGoals": 1}, event_minutes={"goal": ["<b>1'</b>"]}))
    svg = pitch.pitch_svg(team)
    assert "<b>" not in svg and "&lt;b&gt;" in svg
