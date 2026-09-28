# ⚽ Football Predictor

Predicts football match outcomes (1X2, over/under, correct score) using
a [Dixon-Coles](https://en.wikipedia.org/wiki/Dixon%E2%80%93Coles_model) model
(Poisson goal model with a low-score correlation adjustment) fit on historical
results, adjusted for current injuries. Local-only, single-user.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with API keys (all optional for the historical-data-only
workflow below, required for live fixtures/injuries/chat notes):

- `FOOTBALL_DATA_ORG_API_KEY` — free registration at https://www.football-data.org/client/register
- `API_FOOTBALL_KEY` — optional, free tier at https://rapidapi.com/api-sports/api/api-football (weak/partial injury coverage; see "Known limitations" below)
- `ANTHROPIC_API_KEY` — required for the dashboard's "Notes" chat tab, get one at https://console.anthropic.com/settings/keys

If you already have a database from before the chat-notes feature, run the
one-time migration first: `uv run python scripts/migrate_chat_notes.py`.
Same for the team-crest column: `uv run python scripts/migrate_team_crest.py`.

## Usage

Four leagues are configured out of the box: EPL, La Liga, Bundesliga, Serie A
(`config/leagues.yaml`).

1. **Download historical results and fit the model** (no API key needed):

   ```bash
   uv run python scripts/fetch_historical_data.py EPL LALIGA BUNDESLIGA SERIEA
   uv run python scripts/run_training.py EPL LALIGA BUNDESLIGA SERIEA
   ```

   The training script prints a team strength leaderboard per league —
   sanity-check that it roughly matches the real table before trusting
   predictions. Check `data/unmatched_teams.log` afterward; a name that
   didn't resolve is logged there rather than silently guessed (see
   "Adding a league" below).

2. **Pull live fixtures and injuries** (needs API keys from `.env`):

   ```bash
   uv run python scripts/refresh_live_data.py EPL
   ```

   Also fetches each team's crest and each league's emblem for the
   dashboard (from the same football-data.org response, no extra calls).

3. **Launch the dashboard**:

   ```bash
   uv run streamlit run src/soccer_predictor/dashboard/app.py --server.address localhost
   ```

   `--server.address localhost` matters: Streamlit's own default is to
   listen on every network interface (`0.0.0.0`), not just this machine --
   without it, anyone else on the same Wi-Fi/LAN can open your dashboard in
   their own browser and use it, spending your API-Football/football-data.org/
   Anthropic quota. This app has no login of its own.

**For everyday use, just run step 3 on its own.** The dashboard only ever
reads `data/soccer.db` -- it never re-fetches or re-trains anything at
startup, so this is instant no matter how much history is in there. Steps
1-2 are for updating data, not something to re-run before every launch.

When you do want fresh data, scope steps 1-2 to the league(s) you're
actually using, the same way the examples above do (`EPL LALIGA BUNDESLIGA
SERIEA`, not every configured league). Running `refresh_live_data.py` with
**no arguments** syncs *all* configured leagues, including UEFA Champions
League (36 teams) and the World Cup (48 teams) -- each team costs a
rate-limited API call (10/min), so an all-leagues refresh can take several
minutes and isn't something to run routinely. Also run these as separate
commands rather than pasting several at once: a shell queues pasted commands
one after another, so a slow `refresh_live_data.py` run silently delays
`streamlit run` from starting at all, which looks like the dashboard itself
is hanging when it's really just next in line.

   Opens on the **Leagues** page: League and Season selectors at the top
   (with each league's flag/emblem), an **Upcoming Fixtures** section showing
   every scheduled match league-wide for the next 14 days with model
   predictions (not live in-play scores — this app has no real-time feed,
   only the schedule plus our own prediction), and a real league table below
   — Pos/Zone/Team/Pl/W/D/L/GF/GA/GD/Pts/Form/Next, computed straight from
   actual results (`model/standings.py`), independent of the Dixon-Coles
   model. The **Zone** column marks Champions League qualification (green),
   relegation playoff where applicable (amber, Bundesliga only), and
   relegation (red) per `config/leagues.yaml`'s `zones` for that league —
   deliberately simplified (real Europa/Conference League qualification also
   depends on domestic cup winners and varies year to year, so only the
   stable Champions League/relegation bands are marked). All 4 leagues here
   are top-flight divisions, so there's no promotion-playoff zone (that's a
   second-tier thing, e.g. the EFL Championship playoffs — it doesn't apply
   to the Premier League itself). Click any team's row to open **Team
   Detail** (crest, the model's attack/defense rating, upcoming schedule
   with predictions, recent results, current squad, historical player
   stats, injuries) — the model rating only shows up here, not on the main
   table. Two separate player sections, deliberately not merged:
   - **Squad**: the actual current roster (`ingest/squad.py`), from
     football-data.org's same API in one call per league (cached a day),
     matched to our teams via the existing alias table — no new API key or
     DB migration needed. Name/position/nationality/age only.
   - **Player Stats (Historical)**: goals/assists/saves/tackles/cards/rating
     from API-Football (`API_FOOTBALL_KEY`, free tier: 100 req/day, cached
     30 days since a past season never changes), with its own season picker
     (2022/23-2024/25 — the free plan blocks the current season entirely).
     This is intentionally a separate lookup, not fused onto the current
     squad: mixing "who's on the team now" with "last season's stats" would
     mean new signings misleadingly show zeros, so instead you pick a
     season and see whoever actually played for the team that year, under
     API-Football's own player names (`ingest/player_stats.py` — team names
     are fuzzy-matched between the two APIs, since there's no shared id and
     API-Football's own search has zero fuzzy tolerance server-side).

   The original single-league workflow (fixtures table, full
   attack/defense leaderboard, injuries, chat notes, custom matchup) lives
   under **Predictor** in the sidebar, unchanged — reaching it via a
   `?league=` link pre-selects that league.

Re-run `run_training.py` whenever new results come in (roughly weekly) — the
dashboard only reads the latest persisted fit, it never trains live.

## Chat notes (AI-assisted)

The "Notes" tab lets you type freeform notes — e.g. "Saka is injured,
hamstring, ~3 weeks out" or "Arsenal have been flat since the manager
change" — and an AI (Claude Haiku via the Anthropic API) parses them into
structured entries:

- **Player availability** (injury/suspension): adjusts the player's team's
  attack or defense rating via the same mechanism as `config/injuries.yaml`,
  but with an estimated return date — the adjustment automatically stops
  applying once a predicted fixture's date is past that return date.
- **Team form**: a much smaller, explicitly-labeled-as-speculative nudge
  (`model/form_adjustment.py`, capped at ±15%) for notes that don't map to a
  specific player, with its own estimated expiry.

Nothing is saved or affects predictions until you review the parsed entry
in its card and click Save — the AI's output is never applied automatically.
Delete a note any time from the "Active chat-sourced notes" list (e.g. if a
player recovers early or you made a mistake).

## Adding a league

Add an entry to `config/leagues.yaml` with its
[football-data.co.uk](https://www.football-data.co.uk/data.php) CSV code and
[football-data.org](https://www.football-data.org/documentation/api) competition
id, then seed `config/team_aliases.yaml` with that league's clubs (canonical
name + the CSV spelling + the API spelling + `league:` code). Unresolved names
during ingestion are logged to `data/unmatched_teams.log` instead of being
silently guessed — check that file after adding a league or after each
promotion/relegation season and fill in the missing aliases. The La
Liga/Bundesliga/Serie A rows were a good-faith first pass and ingested clean
(zero unmatched names across all 4 seasons each), but non-English spellings
are inherently less certain than EPL's — re-check after each season rollover.

Every alias row's `league:` field matters: `ingest/team_mapper.py::seed_teams_and_aliases`
only processes rows for the league currently being seeded, so unrelated
leagues' clubs never get cross-tagged even though they all live in one file.

## Dashboard structure

`dashboard/app.py` is a thin `st.navigation()`/`st.Page()` entrypoint;
`dashboard/navigation.py` builds the page objects; each page's actual content
is in `dashboard/views/{home,team_detail,predictor}.py`. That
directory is deliberately named `views/`, not `pages/` — Streamlit reserves
`pages/` for its own legacy auto-discovery and will inject extra nav entries
if a directory with that exact name sits next to the entrypoint script, even
when you're driving navigation manually via `st.Page`/`st.navigation`.

## Running tests

```bash
uv run pytest
```

## Known limitations

- **Team-name matching** across the CSV and API sources is ongoing manual
  upkeep (see `config/team_aliases.yaml` above), not a one-time fix.
- **Injury data is weak on free tiers.** `config/injuries.yaml` (manual entry)
  is the reliable source and always overrides the API; the API-Football
  fetch (`ingest/injuries.py`) is best-effort and uses a flat importance
  weight since the endpoint doesn't expose a real importance signal.
- **The injury adjustment's magnitude is a tuned heuristic**
  (`model/injury_adjustment.py`), not empirically validated — there isn't
  enough free injury data to backtest it properly.
- **The form-note adjustment is more speculative still**
  (`model/form_adjustment.py`) — unlike the injury adjustment, even the
  *existence* of an effect from a general note (not tied to a specific
  missing player) is a guess, not just its size. Treat it as a small,
  capped nudge, not a modeled signal.
- **Chat note parsing costs a (tiny) amount per submission** — each note
  is one live Anthropic API call, not cached or batched.
- **Emblems/crests require a live sync.** League emblems and team crests are
  sourced from football-data.org; without `FOOTBALL_DATA_ORG_API_KEY` set (or
  before the first `refresh_live_data.py` run), the dashboard falls back to a
  placeholder icon rather than erroring — historical-only teams never picked
  up in a live sync stay placeholder-only.
- Free-tier API rate limits (football-data.org: 10 req/min; API-Football:
  10 req/min **and** 100 req/day) are respected via on-disk caching in
  `data/cache/` — don't delete that directory and immediately re-run
  refreshes in a loop. Player-stats lookups are cached 30 days (a past
  season never changes), so browsing the same team repeatedly costs
  nothing after the first look.
- **Historical player stats never cover the current season** — API-Football's
  free plan blocks `/players` access to it entirely (verified: it only
  allows 2022-2024). `ingest/player_stats.py::AVAILABLE_SEASONS` is that
  list; extend it if you upgrade the plan.
- **The current in-progress season needs periodic re-fetching to stay
  current.** football-data.co.uk updates the *last* season in each league's
  `seasons` list continuously as real games are played (it's not just a
  historical archive), so `scripts/fetch_historical_data.py` always
  re-downloads that one season's CSV (every earlier season is treated as
  concluded and only ever fetched once) — re-run it periodically (and
  `run_training.py` after) to pick up new results.
