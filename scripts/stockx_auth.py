"""
One-time interactive StockX OAuth login — captures this project's own
refresh_token (ported from sneaker-arbitrage/scripts/stockx_auth.py; see
app/stockx_client.py's module docstring for why this project uses its own
token rather than sharing that app's).

Prereqs (once, at https://developer.stockx.com):
  1. Create an app (or reuse the same one sneaker-arbitrage uses); note its
     client ID, client secret, and API key.
  2. Register this project's redirect URI (STOCKX_REDIRECT_URI in .env,
     default http://localhost:8018/stockx/callback — a different port than
     sneaker-arbitrage's 8017 so both can be registered on the same app).
  3. Put STOCKX_CLIENT_ID / STOCKX_CLIENT_SECRET / STOCKX_API_KEY in .env.

Then:  python scripts/stockx_auth.py

Opens the StockX login page, catches the redirect on a temporary local HTTP
server, verifies the CSRF state, exchanges the code for tokens, and saves
the refresh token to data/stockx_token.json (app/stockx_client.py reads it
from there and keeps it updated across rotations).
"""
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                         # noqa: E402
from app.stockx_client import STOCKX_AUDIENCE, STOCKX_AUTHORIZE_URL, STOCKX_TOKEN_URL  # noqa: E402

_result: dict = {}


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        expected_path = urlparse(settings.stockx_redirect_uri).path or "/"
        if parsed.path != expected_path:
            self.send_response(404)
            self.end_headers()
            return
        qs = parse_qs(parsed.query)
        _result["code"] = (qs.get("code") or [None])[0]
        _result["state"] = (qs.get("state") or [None])[0]
        _result["error"] = (qs.get("error_description") or qs.get("error") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>StockX login captured &mdash; you can close this tab.</h2>")
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, *args):
        pass


def main():
    if not (settings.stockx_client_id and settings.stockx_client_secret):
        sys.exit("STOCKX_CLIENT_ID / STOCKX_CLIENT_SECRET missing from .env — see this script's docstring.")

    state = secrets.token_urlsafe(24)
    authorize_url = STOCKX_AUTHORIZE_URL + "?" + urlencode({
        "response_type": "code",
        "client_id": settings.stockx_client_id,
        "redirect_uri": settings.stockx_redirect_uri,
        "scope": "offline_access openid",
        "audience": STOCKX_AUDIENCE,
        "state": state,
    })

    redirect = urlparse(settings.stockx_redirect_uri)
    server = HTTPServer((redirect.hostname or "localhost", redirect.port or 80), _CallbackHandler)

    print(f"Listening on {settings.stockx_redirect_uri}")
    print("Opening StockX login page (copy the URL below into a browser if it doesn't open):\n")
    print(f"  {authorize_url}\n")
    webbrowser.open(authorize_url)
    server.serve_forever()

    if _result.get("error"):
        sys.exit(f"StockX returned an error: {_result['error']}")
    if _result.get("state") != state:
        sys.exit("CSRF state mismatch — aborting without exchanging the code.")
    code = _result.get("code")
    if not code:
        sys.exit("No authorization code received.")

    print("Exchanging authorization code for tokens …")
    resp = httpx.post(STOCKX_TOKEN_URL, data={
        "grant_type": "authorization_code",
        "client_id": settings.stockx_client_id,
        "client_secret": settings.stockx_client_secret,
        "code": code,
        "redirect_uri": settings.stockx_redirect_uri,
    }, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        sys.exit(f"Token response had no refresh_token (got keys: {list(payload)}). "
                 "Is the 'offline_access' scope enabled for your app?")

    import json
    from datetime import datetime, timedelta
    token_path = Path(settings.stockx_token_cache_path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    expires_at = None
    if payload.get("expires_in"):
        expires_at = (datetime.utcnow() + timedelta(seconds=int(payload["expires_in"]))).isoformat()
    token_path.write_text(json.dumps({
        "refresh_token": refresh_token,
        "access_token": payload.get("access_token", ""),
        "access_token_expires_at": expires_at,
    }), encoding="utf-8")

    print(f"\n✓ Refresh token saved to {token_path} — app/stockx_client.py will")
    print("  refresh access tokens automatically from here on.")


if __name__ == "__main__":
    main()
