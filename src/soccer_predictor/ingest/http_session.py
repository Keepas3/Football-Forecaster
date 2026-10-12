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

_session = requests.Session()
_session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
# Only a connection that could not be made is retried (twice, quickly); a
# request that reached the server is never repeated, so a POST can't run twice.
_adapter = HTTPAdapter(
    pool_connections=8,
    pool_maxsize=16,  # enough for the live banner's parallel calls
    max_retries=Retry(total=None, connect=2, read=0, status=0, other=0, backoff_factor=0.2),
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def get(url: str, **kwargs) -> requests.Response:
    return _session.get(url, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    return _session.post(url, **kwargs)
