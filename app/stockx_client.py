"""
Minimal official StockX API client (developer.stockx.com) — ported and
trimmed from sneaker-arbitrage's app/scrapers/stockx_api.py. Same auth model,
same catalog-match strategy (style-code equality, then name+colorway fuzzy
fallback), same market-data response handling. See that file's docstring for
the verified response shapes; the notes aren't repeated here.

Deliberately simplified vs. the original for this project's much lower,
on-demand call volume (one identify request -> a couple of API calls, not a
bulk multi-thousand-SKU scrape):
  - Refresh-token persistence is a local JSON file (data/stockx_token.json),
    not a DB table — this project has no database of its own.
  - The daily-budget counter is in-memory only (resets on restart), not a
    DB-backed counter shared across processes. Fine at this call volume;
    NOT a substitute for the real per-account cap sneaker-arbitrage enforces
    if you run both against the same StockX account at real scrape volume.
"""
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import httpx

from app.config import settings
from app.models import MarketData, SizeMarket

logger = logging.getLogger(__name__)

STOCKX_API_BASE = "https://api.stockx.com/v2"
STOCKX_TOKEN_URL = "https://accounts.stockx.com/oauth/token"
STOCKX_AUTHORIZE_URL = "https://accounts.stockx.com/authorize"
STOCKX_AUDIENCE = "gateway.stockx.com"

# Same documented defaults as sneaker-arbitrage — see that repo's README
# §StockX API for sourcing. Verify against YOUR developer-portal dashboard.
REQUESTS_PER_SECOND = 1.0
DAILY_REQUEST_LIMIT = 25_000
DAILY_SAFETY_MARGIN = 500

FUZZY_MATCH_THRESHOLD = 0.60
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0

_ALNUM_RE = re.compile(r"[^A-Z0-9]")


class StockXBudgetExhausted(RuntimeError):
    pass


class StockXRequestFailed(RuntimeError):
    def __init__(self, path: str, status: "int | None", detail: str, attempts: int):
        self.path, self.status, self.detail, self.attempts = path, status, detail, attempts
        label = f"HTTP {status}" if status is not None else "transport error"
        super().__init__(f"{label} on {path} after {attempts} attempt(s): {detail}")


def is_configured() -> bool:
    return bool(settings.stockx_client_id and settings.stockx_client_secret
                and settings.stockx_api_key
                and (settings.stockx_refresh_token or Path(settings.stockx_token_cache_path).exists()))


def _norm_style(s: str) -> str:
    return _ALNUM_RE.sub("", (s or "").upper())


def _style_segments(style_id: str) -> List[str]:
    out = []
    for seg in (style_id or "").split("/"):
        seg = re.sub(r"\([^)]*\)", " ", seg)
        norm = _norm_style(seg)
        if norm:
            out.append(norm)
    return out


def _style_matches(style_id: str, target_norm: str) -> bool:
    return any(seg == target_norm for seg in _style_segments(style_id))


def _amount(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class _RateLimiter:
    """Per-second pacing (thread lock) + in-memory daily counter. See the
    module docstring for why this isn't DB-backed like sneaker-arbitrage's."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_at = 0.0
        self._day = ""
        self._calls_today = 0

    def acquire(self):
        with self._lock:
            wait = (1.0 / REQUESTS_PER_SECOND) - (time.monotonic() - self._last_at)
            if wait > 0:
                time.sleep(wait)
            self._last_at = time.monotonic()

            today = datetime.utcnow().strftime("%Y-%m-%d")
            if today != self._day:
                self._day, self._calls_today = today, 0
            self._calls_today += 1
            if self._calls_today > DAILY_REQUEST_LIMIT - DAILY_SAFETY_MARGIN:
                raise StockXBudgetExhausted(
                    f"StockX daily budget exhausted (in-process count): "
                    f"{self._calls_today}/{DAILY_REQUEST_LIMIT}"
                )


_limiter = _RateLimiter()


@dataclass
class _TokenCache:
    refresh_token: str = ""
    access_token: str = ""
    access_token_expires_at: Optional[datetime] = None


class _TokenManager:
    """Refreshes access tokens from a refresh token, persisting both (StockX
    may rotate the refresh token on use) to a local JSON file so a restart
    doesn't force a fresh interactive login."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache = self._load()

    def _load(self) -> _TokenCache:
        path = Path(settings.stockx_token_cache_path)
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                expires = raw.get("access_token_expires_at")
                return _TokenCache(
                    refresh_token=raw.get("refresh_token", ""),
                    access_token=raw.get("access_token", ""),
                    access_token_expires_at=datetime.fromisoformat(expires) if expires else None,
                )
            except (json.JSONDecodeError, ValueError, OSError):
                logger.warning(f"Couldn't read {path} — ignoring cached token.")
        return _TokenCache(refresh_token=settings.stockx_refresh_token)

    def _save(self):
        path = Path(settings.stockx_token_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "refresh_token": self._cache.refresh_token,
            "access_token": self._cache.access_token,
            "access_token_expires_at": (self._cache.access_token_expires_at.isoformat()
                                        if self._cache.access_token_expires_at else None),
        }), encoding="utf-8")

    def get_access_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            c = self._cache
            if (not force_refresh and c.access_token and c.access_token_expires_at
                    and datetime.utcnow() < c.access_token_expires_at - timedelta(seconds=60)):
                return c.access_token

            refresh_token = c.refresh_token or settings.stockx_refresh_token
            if not refresh_token:
                raise RuntimeError(
                    "No StockX refresh token — run scripts/stockx_auth.py once "
                    "to complete the interactive login."
                )

            resp = httpx.post(STOCKX_TOKEN_URL, data={
                "grant_type": "refresh_token",
                "client_id": settings.stockx_client_id,
                "client_secret": settings.stockx_client_secret,
                "refresh_token": refresh_token,
                "audience": STOCKX_AUDIENCE,
            }, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            c.access_token = payload["access_token"]
            expires_in = int(payload.get("expires_in", 43200))
            c.access_token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            if payload.get("refresh_token"):
                c.refresh_token = payload["refresh_token"]
            elif not c.refresh_token:
                c.refresh_token = refresh_token
            self._save()
            logger.info(f"StockX access token refreshed (expires {c.access_token_expires_at:%Y-%m-%d %H:%M} UTC)")
            return c.access_token


_token_manager = _TokenManager()


class StockXAPIClient:
    def __init__(self):
        self._http = httpx.Client(
            base_url=STOCKX_API_BASE,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._shape_warned: set = set()

    def close(self):
        self._http.close()

    def _request(self, path: str, params: Optional[dict] = None) -> "dict | list | None":
        attempt = 0
        last_status, last_detail = None, ""
        while True:
            _limiter.acquire()
            headers = {
                "Authorization": f"Bearer {_token_manager.get_access_token()}",
                "x-api-key": settings.stockx_api_key,
            }
            try:
                resp = self._http.get(path, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_status, last_detail, resp = None, repr(exc), None
            else:
                last_status = resp.status_code
                if resp.status_code == 401 and attempt == 0:
                    _token_manager.get_access_token(force_refresh=True)
                    attempt += 1
                    continue
                if resp.status_code < 400:
                    try:
                        return resp.json()
                    except ValueError:
                        return None
                last_detail = resp.text[:200]

            if last_status not in RETRYABLE_STATUSES and last_status is not None:
                raise StockXRequestFailed(path, last_status, last_detail, attempt + 1)
            if attempt >= MAX_RETRIES:
                raise StockXRequestFailed(path, last_status, last_detail, attempt + 1)
            delay = min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_CAP_SECONDS)
            logger.warning(f"StockX transient failure on {path} (attempt {attempt + 1}), retrying in {delay:.1f}s")
            time.sleep(delay)
            attempt += 1

    def _warn_shape(self, tag: str, payload):
        if tag in self._shape_warned:
            return
        self._shape_warned.add(tag)
        keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        logger.warning(f"[stockx shape] {tag}: unexpected response shape (top-level: {keys})")

    def _search(self, query: str, page_size: int = 10) -> List[dict]:
        data = self._request("/catalog/search", {"query": query, "pageNumber": 1, "pageSize": page_size})
        if isinstance(data, dict) and isinstance(data.get("products"), list):
            return data["products"]
        if data is not None:
            self._warn_shape("catalog/search", data)
        return []

    def _match_product(self, sku: str, name: str = "") -> "tuple[Optional[dict], Optional[str]]":
        """Returns (product, failure_reason)."""
        target = _norm_style(sku)
        products = self._search(sku)
        for p in products:
            if _style_matches(p.get("styleId", ""), target):
                return p, None

        if name:
            candidates = products or self._search(name)
            best, best_ratio = None, 0.0
            for p in candidates:
                colorway = (p.get("productAttributes") or {}).get("colorway") or ""
                label = f"{p.get('title', '')} {colorway}".strip().lower()
                ratio = SequenceMatcher(None, name.lower(), label).ratio()
                if ratio > best_ratio:
                    best, best_ratio = p, ratio
            if best is not None and best_ratio >= FUZZY_MATCH_THRESHOLD:
                return best, None

        return None, ("no_search_results" if not products else "no_style_match")

    def _get_variants(self, product_id: str) -> dict:
        """variantId -> US size label."""
        data = self._request(f"/catalog/products/{product_id}/variants")
        out = {}
        if isinstance(data, list):
            for v in data:
                if not (isinstance(v, dict) and v.get("variantId")):
                    continue
                conv = (v.get("sizeChart") or {}).get("defaultConversion") or {}
                size = str(conv.get("size") or v.get("variantValue") or v.get("variantName") or "")
                if str(conv.get("type") or "").lower() == "us w" and size and not size.upper().startswith("W"):
                    size = f"W{size}"
                out[v["variantId"]] = size
        elif data is not None:
            self._warn_shape("catalog/variants", data)
        return out

    def get_market(self, sku: str, name: str = "") -> "tuple[Optional[MarketData], Optional[str]]":
        """Full lookup: catalog match -> per-variant market data.
        Returns (MarketData, None) on success, (None, failure_reason) otherwise.
        Never raises — infra/auth failures are caught and reported as a
        failure_reason instead, same as sneaker-arbitrage's get_market()
        (a caller shouldn't have to try/except an identify request just
        because credentials aren't set up yet or StockX hiccuped)."""
        try:
            return self._get_market(sku, name)
        except StockXBudgetExhausted as exc:
            return None, f"error:budget_exhausted:{exc}"
        except StockXRequestFailed as exc:
            status = exc.status if exc.status is not None else "transport"
            return None, f"error:{status}:{exc.path}"
        except Exception as exc:
            logger.exception(f"StockX lookup failed for {sku} (unexpected)")
            return None, f"error:{type(exc).__name__}:{exc}"

    def _get_market(self, sku: str, name: str) -> "tuple[Optional[MarketData], Optional[str]]":
        product, why = self._match_product(sku, name)
        if product is None:
            return None, why

        product_id = product.get("productId") or product.get("id")
        if not product_id:
            self._warn_shape("catalog/search:productId", product)
            return None, "error:no_productId"

        variants = self._get_variants(product_id)
        market = self._request(f"/catalog/products/{product_id}/market-data", {"currencyCode": "USD"})
        rows = market if isinstance(market, list) else []
        if not rows and isinstance(market, dict):
            for key in ("variants", "marketData", "data"):
                if isinstance(market.get(key), list):
                    rows = market[key]
                    break
        if not rows and market is not None and not isinstance(market, list):
            self._warn_shape("catalog/market-data", market)

        sizes: List[SizeMarket] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            size = variants.get(row.get("variantId"), "") or str(row.get("variantValue") or "ANY")
            std = row.get("standardMarketData") or {}
            lowest_ask = _amount(row.get("lowestAskAmount")) or _amount(std.get("lowestAsk"))
            highest_bid = _amount(row.get("highestBidAmount")) or _amount(std.get("highestBidAmount"))
            if lowest_ask is None and highest_bid is None:
                continue
            sizes.append(SizeMarket(size=size, lowest_ask=lowest_ask, highest_bid=highest_bid))

        if not sizes:
            return None, "no_market_data"

        url_key = product.get("urlKey") or ""
        overall_ask = min((s.lowest_ask for s in sizes if s.lowest_ask is not None), default=None)
        overall_bid = max((s.highest_bid for s in sizes if s.highest_bid is not None), default=None)
        return MarketData(
            product_id=str(product_id),
            url=f"https://stockx.com/{url_key}" if url_key else "https://stockx.com",
            lowest_ask=overall_ask,
            highest_bid=overall_bid,
            sizes=sizes,
        ), None
