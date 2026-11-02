"""One shared HTTP session for the API clients, so connections are reused.

Every `requests.get` opens a brand-new TLS connection, which is most of a
request's cost: measured against ESPN, 8 scoreboard calls took 5.3s that way
(0.66s each) and 0.94s (0.12s each) on one kept-alive session. A refresh makes
hundreds of calls, so this is the biggest single speed-up available.

Cookies are switched off. These are all anonymous API/RSS reads, nothing needs
a cookie, and a session shared by every dashboard viewer must not carry one
visitor's state (or a consent cookie some site sets) over to another's request.
"""

from __future__ import annotations

from http.cookiejar import DefaultCookiePolicy

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _new_session(retry: Retry | int) -> requests.Session:
    session = requests.Session()
    session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
    adapter = HTTPAdapter(
        pool_connections=8,
        pool_maxsize=16,  # enough for the live banner's parallel calls
        max_retries=retry,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Only a connection that could not be made is retried (twice, quickly); a
# request that reached the server is never repeated, so a POST can't run twice.
_session = _new_session(Retry(total=None, connect=2, read=0, status=0, other=0, backoff_factor=0.2))
# For a request a page waits on whose source may be down: with the retries above a host that
# never answers costs three connect timeouts before the failure is reported.
_single_attempt_session = _new_session(0)


def get(url: str, **kwargs) -> requests.Response:
    return _session.get(url, **kwargs)


def post(url: str, single_attempt: bool = False, **kwargs) -> requests.Response:
    """`single_attempt` skips the connection retries: the first failure is the answer."""
    return (_single_attempt_session if single_attempt else _session).post(url, **kwargs)
