"""
Swappable web/image search providers. The pipeline only depends on the
SearchProvider protocol; Serper.dev is the first implementation.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol
from urllib.parse import urlparse

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class SearchError(Exception):
    pass


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str = ""

    @property
    def domain(self) -> str:
        return (urlparse(self.url).hostname or "").lower().removeprefix("www.")


@dataclass
class ImageResult:
    image_url: str
    source_url: str = ""
    title: str = ""
    width: int = 0
    height: int = 0

    @property
    def domain(self) -> str:
        return (urlparse(self.source_url or self.image_url).hostname or "").lower().removeprefix("www.")


class SearchProvider(Protocol):
    name: str
    available: bool

    def web_search(self, query: str, num: int = 10) -> List[WebResult]: ...
    def image_search(self, query: str, num: int = 10) -> List[ImageResult]: ...


class NullProvider:
    name = "none"
    available = False

    def web_search(self, query: str, num: int = 10) -> List[WebResult]:
        raise SearchError("no search provider configured")

    def image_search(self, query: str, num: int = 10) -> List[ImageResult]:
        raise SearchError("no search provider configured")


class SerperProvider:
    name = "serper"
    _BASE = "https://google.serper.dev"

    def __init__(self, api_key: Optional[str] = None, client: Optional[httpx.Client] = None):
        self._key = api_key or settings.serper_api_key
        self._http = client or httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))

    @property
    def available(self) -> bool:
        return bool(self._key)

    def _post(self, path: str, query: str, num: int) -> dict:
        if not self._key:
            raise SearchError("SERPER_API_KEY not set")
        try:
            resp = self._http.post(f"{self._BASE}{path}", json={"q": query, "num": num, "gl": "us", "hl": "en"},
                                   headers={"X-API-KEY": self._key, "Content-Type": "application/json"})
        except httpx.HTTPError as exc:
            raise SearchError(f"transport:{exc!r}") from exc
        if resp.status_code >= 400:
            raise SearchError(f"http_{resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise SearchError("bad_json") from exc

    def web_search(self, query: str, num: int = 10) -> List[WebResult]:
        data = self._post("/search", query, num)
        out = []
        for item in data.get("organic") or []:
            if item.get("link"):
                out.append(WebResult(title=item.get("title") or "", url=item["link"],
                                     snippet=item.get("snippet") or ""))
        kg = data.get("knowledgeGraph") or {}
        if kg.get("title"):
            desc = " ".join(f"{k}: {v}" for k, v in (kg.get("attributes") or {}).items())
            out.insert(0, WebResult(title=kg["title"], url=kg.get("website") or kg.get("descriptionLink") or "",
                                    snippet=f"{kg.get('description') or ''} {desc}".strip()))
        return out[:num]

    def image_search(self, query: str, num: int = 10) -> List[ImageResult]:
        data = self._post("/images", query, num)
        out = []
        for item in data.get("images") or []:
            if item.get("imageUrl"):
                out.append(ImageResult(image_url=item["imageUrl"], source_url=item.get("link") or "",
                                       title=item.get("title") or "",
                                       width=int(item.get("imageWidth") or 0), height=int(item.get("imageHeight") or 0)))
        return out[:num]


_provider: Optional[SearchProvider] = None


def get_provider() -> SearchProvider:
    global _provider
    if _provider is None:
        if settings.search_provider == "serper" and settings.serper_api_key:
            _provider = SerperProvider()
        else:
            _provider = NullProvider()
    return _provider


def set_provider(provider: Optional[SearchProvider]) -> None:
    """Test hook / dependency injection."""
    global _provider
    _provider = provider
