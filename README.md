# Football Match Predictor

A local, single-user Streamlit app that predicts football match outcomes
(win/draw/loss, over/under, correct score) using a
[Dixon-Coles](https://en.wikipedia.org/wiki/Dixon%E2%80%93Coles_model) model
(a Poisson goal model with a low-score correlation adjustment) fit on
historical results, adjusted for current injuries and team form.

## What it does

- **Predicts match outcomes** for six leagues with enough historical data to
  train from: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, and
  MLS. Each prediction shows the full probability breakdown (home win,
  draw, away win, over/under 2.5 goals, both teams to score, most likely
  scorelines) and the worked Poisson/Dixon-Coles math behind it.
- **Tracks its own accuracy.** Every prediction is locked in the moment a
  fixture is seen as upcoming, before the result is known. Once the real
  result comes in, that locked-in prediction is graded as an exact score
  match, a correct win/draw/loss call with the wrong score, or wrong
  entirely. Running counters and a hit rate are shown in the Predictor
  page's sidebar.
- **Shows live fixtures, results, and standings** for three additional
  competitions with no historical model attached (UEFA Champions League,
  European Championship, World Cup), since those either lack enough
  matching historical data or don't run on a normal weekly season.
- **Adjusts predictions for injuries**, from three merged sources: a manual
  YAML file (most trusted), a best-effort API fetch, and freeform notes
  typed into a chat tab that an AI parses into structured entries. Injuries
  reduce the injured player's team's attack or defense rating in proportion
  to how important that player actually is, based on their own real
  goals, assists, and minutes played.
- **Lets you search for any player** by name across an entire league (not
  just one team at a time), showing their current club, current-season
  expected goals and assists where available, and historical per-season
  stats on request.
- **Splits standings by conference** for leagues that are actually
  organized that way (MLS's Eastern and Western conferences), instead of
  forcing everything into one flat table.
- **Links match results to YouTube** search results for that specific
  fixture, so a past result can be watched directly from the dashboard.
- Runs entirely against a local SQLite database. The dashboard itself never
  makes a live network call on page load; it only reads what was already
  fetched and cached by the ingestion scripts.

## Data sources

- **football-data.co.uk**: historical match results (CSV) for the five CSV-
  backed domestic leagues. This is what the Dixon-Coles model is trained
  from for those leagues.
- **football-data.org**: live fixtures, current-season results, squads, and
  team/league emblems for every league except MLS.
- **ESPN's public site API** (undocumented, no key required): fixtures,
  results, squads, and multi-season historical results for MLS, which
  football-data.org does not cover at all.
- **Understat** (undocumented, no key required): historical and current-
  season goals/assists/xG/xA/shots/key passes/cards for the five European
  leagues it tracks, back to 2014. Used on the Players/Team Detail pages
  and to size injury adjustments more accurately than raw goals/assists
  alone.
- **American Soccer Analysis** (documented, no key required): the same
  role as Understat, for MLS specifically -- goals/assists/xG/xA/shots/key
  passes/points added, back to 2013.
- **Anthropic (Claude)**: parses freeform chat notes into structured
  injury or form entries. Optional; the rest of the app works without it.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with API keys. All are optional for the historical-data-only
workflow below; `FOOTBALL_DATA_ORG_API_KEY` is required for live fixtures
and chat notes on most leagues:

- `FOOTBALL_DATA_ORG_API_KEY`: free registration at https://www.football-data.org/client/register
- `ANTHROPIC_API_KEY`: required for the dashboard's "Notes" chat tab, get one at https://console.anthropic.com/settings/keys

MLS needs neither key for its live data; it uses ESPN's public API, which
requires no key at all. Understat and American Soccer Analysis (player
stats) are also both keyless.

If you're upgrading an existing database, run any migration scripts you
haven't run yet (each is safe to re-run): `uv run python scripts/migrate_chat_notes.py`,
`migrate_team_crest.py`, `migrate_api_football_team_id.py`,
`migrate_multi_league_teams.py`, `migrate_fixture_kickoff.py`,
`migrate_espn_team_id.py`, `migrate_team_conference.py`.

## Usage

1. **Download historical results and fit the model** (no API key needed
   for the five CSV-backed leagues; MLS needs its ESPN backfill, also
   keyless):

   ```bash
   uv run python scripts/fetch_historical_data.py EPL LALIGA BUNDESLIGA SERIEA LIGUE1 MLS
   uv run python scripts/run_training.py EPL LALIGA BUNDESLIGA SERIEA LIGUE1 MLS
   ```

   The training script prints a team strength leaderboard per league.
   Sanity-check that it roughly matches the real table before trusting
   predictions. Check `data/unmatched_teams.log` afterward; a team name
   that didn't resolve is logged there rather than silently guessed (see
   "Adding a league" below).

2. **Pull live fixtures and injuries** (needs API keys from `.env` for
   every league except MLS):

   ```bash
   uv run python scripts/refresh_live_data.py EPL
   ```

   This also fetches each team's crest and each league's emblem, and locks
   in a prediction snapshot for every upcoming fixture so accuracy can be
   tracked later.

3. **Launch the dashboard**:

   ```bash
   uv run streamlit run src/soccer_predictor/dashboard/app.py --server.address localhost
   ```

   The `--server.address localhost` flag matters. Streamlit's own default
   is to listen on every network interface, not just this machine.
   Without it, anyone else on the same Wi-Fi or LAN could open your
   dashboard in their own browser and use it, spending your API quota.
   This app has no login of its own.

For everyday use, just run step 3 on its own. The dashboard only ever reads
the local database; it never re-fetches or re-trains anything at startup,
so this is instant no matter how much history is in there. Steps 1 and 2
are for updating data, not something to re-run before every launch.

When you do want fresh data, scope steps 1 and 2 to the league(s) you're
actually using, the same way the examples above do. Running
`refresh_live_data.py` with no arguments syncs every configured league,
including UEFA Champions League (36 teams) and the World Cup (48 teams);
each team costs a rate-limited API call, so an all-leagues refresh can take
several minutes and isn't something to run routinely. Also run these as
separate commands rather than pasting several at once: a shell queues
pasted commands one after another, so a slow `refresh_live_data.py` run
silently delays `streamlit run` from starting, which looks like the
dashboard itself is hanging when it's really just next in line.

Re-run `run_training.py` whenever new results come in (roughly weekly).
The dashboard only reads the latest persisted fit; it never trains live.

### Dashboard pages

- **Leagues**: pick a league and season. Shows an upcoming-matches
  section (with predictions where a model exists) and a real league table
  computed straight from actual results, independent of the Dixon-Coles
  model. For MLS, the table splits into Eastern and Western Conference. A
  zone marker highlights Champions League/Concacaf Champions Cup
  qualification and relegation spots where that applies. Click a team's
  row to open Team Detail: crest, the model's attack/defense rating,
  upcoming schedule with predictions, recent results (each linking to a
  YouTube search for that match), current squad, historical per-player
  stats, and injuries.
- **Players**: search for any player by name across a chosen league,
  without needing to know their club first. Shows the matched player's
  current club, position, age, current-season expected goals/assists
  where available, and historical per-season stats on request.
- **Predictor**: the original single-league workflow: a fixtures tab,
  full attack/defense leaderboard, injuries tab, and the chat notes tab.
  Its sidebar also shows the running prediction-accuracy counters
  described above. Reaching it via a `?league=` link pre-selects that
  league.

## Chat notes (AI-assisted)

The "Notes" tab on the Predictor page lets you type freeform notes, for
example "Saka is injured, hamstring, about 3 weeks out" or "Arsenal have
been flat since the manager change", and an AI (Claude Haiku via the
Anthropic API) parses them into structured entries:

- **Player availability** (injury/suspension): adjusts the player's
  team's attack or defense rating the same way `config/injuries.yaml`
  does, but with an estimated return date. The adjustment automatically
  stops applying once a predicted fixture's date is past that return date.
- **Team form**: a much smaller nudge, explicitly labeled as speculative
  (`model/form_adjustment.py`, capped at plus or minus 15 percent), for
  notes that don't map to a specific player, with its own estimated
  expiry.

Nothing is saved or affects predictions until you review the parsed entry
in its card and click Save. The AI's output is never applied
automatically. Delete a note any time from the active chat-sourced notes
list, for example if a player recovers early or you made a mistake.

## Adding a league

For a league backed by a football-data.co.uk CSV: add an entry to
`config/leagues.yaml` with its
[football-data.co.uk](https://www.football-data.co.uk/data.php) CSV code
and [football-data.org](https://www.football-data.org/documentation/api)
competition id, then seed `config/team_aliases.yaml` with that league's
clubs (canonical name, the CSV spelling, the API spelling, and the league
code). Unresolved names during ingestion are logged to
`data/unmatched_teams.log` instead of being silently guessed; check that
file after adding a league or after each promotion/relegation season and
fill in the missing aliases.

Every alias row's league field matters:
`ingest/team_mapper.py::seed_teams_and_aliases` only processes rows for the
league currently being seeded, so unrelated leagues' clubs never get
cross-tagged even though they all live in one file.

For a league with no football-data.co.uk CSV, there are two patterns
already in use, depending on whether football-data.org covers it:

- If football-data.org does cover it (as with UEFA Champions League, the
  European Championship, and the World Cup): omit `csv_code`, set
  `api_competition_id`, and team seeding happens automatically from the
  live API with no alias file entries needed at all. Leave `trainable`
  unset (or false) unless you also want a model fit from whatever match
  history football-data.org can provide.
- If football-data.org doesn't cover it (as with MLS): set
  `data_source: espn` and `espn_league_slug` to that league's slug on
  ESPN's site, and everything (fixtures, results, squads, injuries, team
  seeding) routes through `ingest/espn_client.py` instead. Set
  `trainable: true` if enough historical match data is available to fit a
  real model from (verify this live before assuming it; ESPN's API is
  undocumented and coverage varies by sport and league).

## Dashboard structure

`dashboard/app.py` is a thin `st.navigation()`/`st.Page()` entrypoint;
`dashboard/navigation.py` builds the page objects; each page's actual
content is in `dashboard/views/{home,players,team_detail,predictor}.py`.
That directory is deliberately named `views/`, not `pages/`. Streamlit
reserves `pages/` for its own legacy auto-discovery and will inject extra
navigation entries if a directory with that exact name sits next to the
entrypoint script, even when navigation is being driven manually through
`st.Page`/`st.navigation`.

## Running tests

```bash
uv run pytest
```

## Known limitations

- **Team-name matching** across the CSV and API sources is ongoing manual
  upkeep (see `config/team_aliases.yaml` above), not a one-time fix.
- **Automatic injury detection only exists for MLS** (via ESPN's roster
  data). Every other league has no automatic injury source at all --
  football-data.org has no injuries endpoint, and the API-Football
  integration this app used to have for the other leagues was removed once
  that account became permanently unusable, not just temporarily
  suspended. `config/injuries.yaml` (manual entry) and the AI chat notes
  tab are the real sources of injury data for those leagues; both always
  override anything an automatic source might otherwise supply.
- **The injury adjustment's magnitude is a tuned heuristic**
  (`model/injury_adjustment.py`), not empirically validated. There isn't
  enough free injury data to backtest it properly.
- **The form-note adjustment is more speculative still**
  (`model/form_adjustment.py`). Unlike the injury adjustment, even the
  existence of an effect from a general note not tied to a specific
  missing player is a guess, not just its size. Treat it as a small,
  capped nudge, not a modeled signal.
- **Chat note parsing costs a small amount per submission.** Each note is
  one live Anthropic API call, not cached or batched.
- **Emblems and crests require a live sync.** League emblems and team
  crests are sourced from football-data.org (or ESPN for MLS); without
  the relevant setup, or before the first `refresh_live_data.py` run, the
  dashboard falls back to a placeholder icon rather than erroring.
  Historical-only teams never picked up in a live sync stay
  placeholder-only.
- **football-data.org's free-tier rate limit** (10 requests/minute) is
  respected via on-disk caching in `data/cache/`. Don't delete that
  directory and immediately re-run refreshes in a loop. Understat and
  American Soccer Analysis have no published rate limit, but are still
  called through a self-imposed courtesy limiter and disk cache, since
  both are undocumented/unofficial-ish endpoints that could change or
  start blocking heavy use without notice.
- **Historical player stats only cover 5 European leagues and MLS**
  (Understat and American Soccer Analysis respectively -- both genuinely
  cover the current season too, unlike a typical free-tier stats API).
  UCL/EURO/World Cup have no per-player stats source at all: they pull
  players from dozens of different domestic leagues each, and no free API
  with that cross-league scope was found.
- **The current in-progress season needs periodic re-fetching to stay
  current.** For CSV-backed leagues, football-data.co.uk updates the last
  season in each league's `seasons` list continuously as real games are
  played, so `scripts/fetch_historical_data.py` always re-downloads that
  one season's CSV (every earlier season is treated as concluded and only
  fetched once). For MLS, the same applies through ESPN instead of a CSV.
  Re-run periodically, and `run_training.py` after, to pick up new results.
- **Prediction accuracy tracking only covers predictions made after the
  feature shipped.** A prediction is only graded if it was locked in as a
  forward snapshot before the match was played; there's no way to
  retroactively grade matches that were already finished before this
  feature existed, since no snapshot exists for them.
