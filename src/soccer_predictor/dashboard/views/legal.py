"""The Privacy & Terms page: what the site does with what a visitor gives it, what the forecasts
are and are not, and where the data comes from.

Every statement here describes what the code does (nothing about a visitor is stored in the
database; a what-if note goes to Google's Gemini API; the browser fetches flags and crests from
the hosts below). When a change adds another third party, a new kind of data kept about
visitors, or a new data source, update this page with it -- tests/test_legal_page.py checks the
hosts the code is known to use are named here.
"""

from __future__ import annotations

import html

import streamlit as st

from soccer_predictor.config import load_site_info

LAST_UPDATED = "2 November 2026"

SUMMARY = """
- This is an independent, unofficial football forecasting site. Its forecasts are estimates
  for entertainment, **not betting or financial advice**.
- There are no accounts and no sign-up, and the site does not ask for your name or email. It adds no
  advertising, analytics or tracking of its own.
- If you use the **AI what-if**, the note you type is sent to Google's Gemini API. Please don't
  put personal details in it.
"""

ABOUT = """
Football Forecaster is an independent project. It is not affiliated with, endorsed by or
sponsored by any league, club, competition, player or data provider mentioned on it. Team names,
crests and flags belong to their owners and are shown only to identify the teams and countries.
"""

FORECASTS = """
The scores, win/draw/loss and over/under percentages, ratings and the Track Record come from a
statistical model that learns from past results. They are estimates and will often be wrong, and
how well they did before says nothing certain about what comes next. They are for entertainment
and information only: not betting, gambling, financial or any other advice, and not an offer or
encouragement to bet. Please don't rely on them to place bets; if you gamble, do so within the
law where you live and only with money you can afford to lose.

**How to read them**

- **Early** forecasts are made from the squads and injuries before the lineups are out, so they
  can still change.
- **Locked** forecasts are saved when the starting lineups are announced (about an hour before
  kickoff), or just before kickoff if no lineup is published. Only a locked forecast is graded,
  after the match, and the Track Record counts those.
- **Right outcome** means the model's most likely result (win, draw or loss) happened.
"""

PRIVACY_COLLECT = """
There are no accounts, and the site does not ask for your name, email or any other personal
detail. Nothing about visitors is stored in the site's database, which holds football data,
forecasts and notes added by the site's owner. The site adds no advertising, analytics or
tracking of its own.
"""

PRIVACY_WHATIF = """
The note you type is sent to **Google's Gemini API** (on its free tier), where Google may use it
to improve its products under its own terms, so please don't include personal details. The site
does not save your note or the answer in its database. They stay in your browser session, and are
also held in the server's memory for up to an hour (so the same note on the same match isn't sent
twice) and lost when the server restarts. Use is limited per visit and across all visitors; those
counts are kept in memory and are not tied to your IP address or identity.
"""

PRIVACY_SEARCH = """
Player, club and team searches run on the site's server. A news search sends the words you type to
**Google News**, and the results are kept on the server for a while so the same search isn't
repeated.
"""

PRIVACY_TIME_ZONE = """
You pick your time zone from a list; it is not detected from your device. The choice is kept in
the page address (`tz=...`) so links keep it.
"""

PRIVACY_THIRD_PARTIES = """
To show flags, club crests and a flag font, **your browser loads files directly from other
sites**: {hosts}. When it does, those sites can see your IP address and browser details, under
their own privacy policies. The site also links out to news articles and to YouTube searches; those
sites are not ours.
"""

PRIVACY_HOSTING = """
The site is hosted on **Streamlit Community Cloud**. Its operator handles technical data about
visitors (such as IP address, browser and the pages requested) and may use cookies or similar
storage to run the app; see [Streamlit's privacy policy](https://streamlit.io/privacy-policy).
"""

TERMS = """
By using the site you agree to these terms; if you don't, please don't use it.

- **As is.** The site is provided as it is, with no promise that it is correct, complete, current
  or always available. Its data comes from third-party sources, some of them unofficial, so scores,
  results, injuries, lineups and fixtures can be late, wrong or missing, and can differ from the
  official ones.
- **Reasonable use.** Please don't scrape the site or use it in bulk through automated means, try
  to get around its limits or its password, probe it for weaknesses, or interfere with it. The AI
  what-if shares a free daily allowance between all visitors.
- **Liability.** To the fullest extent the law allows, the site's operator is not liable for any
  loss or damage arising from using the site or relying on it, including anything lost by betting
  or by any other decision made using it. Nothing here limits a liability that the law does not
  allow to be limited.
- **Changes.** The site, its features and these terms may change, or be withdrawn, at any time.
  The date at the top of this page says when it was last updated.
- **Other people's names and content.** Names, logos, crests and data belong to their owners and
  are used under whatever terms they publish.
"""

# What each source supplies, and who. The credits the licences ask for are here too.
SOURCES = """
- **football-data.org**: fixtures, results, squads, top scorers and club crests.
- **football-data.co.uk**: historical results of the domestic leagues.
- **ESPN**: lineups and match details; fixtures and results for competitions football-data.org does
  not cover (such as MLS); and some club crests.
- **Understat**: player statistics for the Premier League, La Liga, Bundesliga, Serie A and Ligue 1.
- **American Soccer Analysis**: MLS player statistics (salaries: MLS Players Association).
- **Fantasy Premier League**: Premier League injury news and defender and goalkeeper statistics.
- **Biwenger** and **Starting 11**: injury news for other leagues.
- **Google News**: news headlines.
- **flagcdn.com**: flags. **Twemoji** country-flag font: graphics by Twitter, Inc. and other
  contributors, licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Wikimedia Commons**: a club crest.

Some of these sources are unofficial and not offered under a data licence, so they can change or be
withdrawn without notice.
"""


def third_party_hosts() -> list[str]:
    """The sites a visitor's browser fetches flags, crests and the flag font from."""
    from urllib.parse import urlparse

    from soccer_predictor.dashboard.components import FLAG_FONT_URL

    return sorted(
        {urlparse(FLAG_FONT_URL).netloc, "flagcdn.com", "crests.football-data.org", "a.espncdn.com", "upload.wikimedia.org"}
    )


def contact_markdown() -> str:
    """The operator and contact lines, or "" when neither is set (config/site.yaml)."""
    site = load_site_info()
    lines = []
    if site.operator:
        lines.append(f"The site is run by {html.escape(site.operator)}.")
    if site.contact_email:
        email = html.escape(site.contact_email)
        lines.append(f"Questions about this page, or anything on the site you want looked at: [{email}](mailto:{email}).")
    return "  \n".join(lines)


def render() -> None:
    left, _ = st.columns([3, 1])  # lines of text this long are hard to read across a wide screen
    with left:
        st.title("Privacy & Terms")
        st.caption(f"Last updated {LAST_UPDATED}")
        st.markdown(SUMMARY)

        st.subheader("About this site")
        st.markdown(ABOUT)

        st.subheader("The forecasts are not betting advice")
        st.markdown(FORECASTS)

        st.subheader("Privacy")
        st.markdown("**What the site collects**")
        st.markdown(PRIVACY_COLLECT)
        st.markdown("**The AI what-if**")
        st.markdown(PRIVACY_WHATIF)
        st.markdown("**Searching**")
        st.markdown(PRIVACY_SEARCH)
        st.markdown("**Time zone**")
        st.markdown(PRIVACY_TIME_ZONE)
        st.markdown("**Content loaded from other sites**")
        st.markdown(PRIVACY_THIRD_PARTIES.format(hosts=", ".join(f"`{host}`" for host in third_party_hosts())))
        st.markdown("**Hosting**")
        st.markdown(PRIVACY_HOSTING)

        st.subheader("Terms of use")
        st.markdown(TERMS)

        st.subheader("Data sources and credits")
        st.markdown(SOURCES)

        contact = contact_markdown()
        if contact:
            st.subheader("Contact")
            st.markdown(contact)
