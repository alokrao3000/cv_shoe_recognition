"""
URL → ProductPageData. Structured data first (JSON-LD / microdata), then
platform JSON (Shopify /products/<handle>.json), then embedded app state
(__NEXT_DATA__ and other inline JSON), then OpenGraph/meta, then visible HTML.
Never relies on a single CSS selector.

Fetches are SSRF-guarded: only http(s), public IPs, bounded size and time.
"""
import ipaddress
import json
import logging
import re
import socket
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

from app.config import settings
from app.extraction import sku_detector
from app.extraction.normalize import canonical_brand
from app.schemas import ProductPageData, StyleCodeCandidate

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"}
_MAX_BYTES = 6 * 1024 * 1024
_MAX_IMAGES = 12

_IDENTIFIER_KEYS = ("sku", "mpn", "stylecode", "style_code", "stylecolor", "style_color", "styleid",
                    "style_id", "productcode", "product_code", "manufacturersku", "manufacturer_sku",
                    "modelnumber", "model_number", "articlenumber", "article_number", "itemnumber",
                    "item_number", "partnumber", "part_number", "style", "colorcode", "color_code", "article")
_IMAGE_KEYS = ("image", "images", "imageurl", "image_url", "src", "url", "imagesrc", "thumbnail", "gallery")
_IMG_EXT_RE = re.compile(r"\.(?:jpe?g|png|webp|avif)(?:\?|$)", re.IGNORECASE)
_IMG_SKIP_RE = re.compile(r"logo|icon|sprite|badge|flag|payment|placeholder|pixel|tracking|avatar|banner|"
                          r"favicon|swatch|1x1|blank|spacer", re.IGNORECASE)


class FetchBlocked(Exception):
    pass


def _assert_public_url(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise FetchBlocked("only http(s) URLs are allowed")
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise FetchBlocked(f"could not resolve {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise FetchBlocked("private/internal addresses are not allowed")
    return parsed.scheme, host


def _get(client: httpx.Client, url: str, accept_json: bool = False) -> Optional[httpx.Response]:
    try:
        headers = dict(_HEADERS)
        if accept_json:
            headers["Accept"] = "application/json"
        with client.stream("GET", url, headers=headers, follow_redirects=True,
                           timeout=settings.fetch_timeout_seconds) as resp:
            if resp.status_code >= 400:
                return None
            chunks, total = [], 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > _MAX_BYTES:
                    break
                chunks.append(chunk)
            resp._content = b"".join(chunks)
            return resp
    except (httpx.HTTPError, ValueError) as exc:
        logger.info(f"fetch failed for {url}: {exc}")
        return None


def fetch_bytes(url: str, client: Optional[httpx.Client] = None) -> Optional[bytes]:
    """Public helper for image downloads with the same guards."""
    try:
        _assert_public_url(url)
    except FetchBlocked:
        return None
    own = client is None
    client = client or httpx.Client()
    try:
        resp = _get(client, url)
        return resp.content if resp is not None else None
    finally:
        if own:
            client.close()


# ── JSON walking helpers ──

def _walk(obj: Any, depth: int = 0) -> Iterable[Tuple[str, Any]]:
    """Yields (key, value) for every dict entry, recursively (bounded depth)."""
    if depth > 12:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
            yield from _walk(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:200]:
            yield from _walk(v, depth + 1)


def _collect_identifiers(obj: Any) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for k, v in _walk(obj):
        kl = k.lower().replace("-", "").replace(" ", "")
        if kl in _IDENTIFIER_KEYS and isinstance(v, (str, int)) and str(v).strip():
            found.setdefault(kl, str(v).strip())
    return found


def _collect_image_urls(obj: Any, base: str) -> List[str]:
    out: List[str] = []
    for k, v in _walk(obj):
        if k.lower() not in _IMAGE_KEYS:
            continue
        vals = v if isinstance(v, list) else [v]
        for item in vals:
            if isinstance(item, dict):
                item = item.get("src") or item.get("url") or item.get("contentUrl") or ""
            if isinstance(item, str) and _IMG_EXT_RE.search(item) and not _IMG_SKIP_RE.search(item):
                out.append(urljoin(base, item.strip()))
    return out


def _dedupe(urls: Iterable[str], limit: int = _MAX_IMAGES) -> List[str]:
    seen, out = set(), []
    for u in urls:
        if not u:
            continue
        if u.startswith("//"):
            u = "https:" + u
        key = u.split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
        if len(out) >= limit:
            break
    return out


def _as_float(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


# ── Structured data ──

def _jsonld_products(html: str) -> List[dict]:
    products, breadcrumbs = [], []
    try:
        import extruct
        data = extruct.extract(html, syntaxes=["json-ld", "microdata"], uniform=True)
    except Exception:
        data = {"json-ld": [], "microdata": []}
        for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                             html, re.IGNORECASE | re.DOTALL):
            try:
                data["json-ld"].append(json.loads(m.group(1)))
            except ValueError:
                continue

    def visit(node):
        if isinstance(node, list):
            for n in node:
                visit(n)
            return
        if not isinstance(node, dict):
            return
        t = node.get("@type")
        types = t if isinstance(t, list) else [t]
        if any(str(x).lower() in ("product", "productmodel", "individualproduct") for x in types if x):
            products.append(node)
        if any(str(x).lower() == "breadcrumblist" for x in types if x):
            breadcrumbs.append(node)
        for key in ("@graph", "mainEntity", "itemListElement", "hasVariant", "isVariantOf"):
            if key in node:
                visit(node[key])

    for syntax in ("json-ld", "microdata"):
        visit(data.get(syntax) or [])
    return products + [{"@type": "BreadcrumbList", **b} for b in breadcrumbs]


def _brand_name(v) -> str:
    if isinstance(v, dict):
        return str(v.get("name") or "")
    if isinstance(v, list) and v:
        return _brand_name(v[0])
    return str(v or "")


def _apply_jsonld(page: ProductPageData, nodes: List[dict], base: str) -> None:
    for node in nodes:
        if str(node.get("@type", "")).lower() == "breadcrumblist":
            items = node.get("itemListElement") or []
            names = []
            for it in items:
                if isinstance(it, dict):
                    name = it.get("name") or (it.get("item") or {}).get("name") if isinstance(it.get("item"), dict) else it.get("name")
                    if name:
                        names.append(str(name))
            if names and not page.breadcrumbs:
                page.breadcrumbs = names
            continue
        page.extraction_sources.append("jsonld")
        page.title = page.title or str(node.get("name") or "")
        page.brand = page.brand or _brand_name(node.get("brand"))
        page.description = page.description or re.sub(r"<[^>]+>", " ", str(node.get("description") or ""))
        page.category = page.category or str(node.get("category") or "")
        for key in ("sku", "mpn", "productID", "gtin13", "gtin12", "gtin", "gtin14", "model"):
            if node.get(key):
                page.raw_identifiers.setdefault(key.lower(), str(node[key]))
        offers = node.get("offers")
        offer = offers[0] if isinstance(offers, list) and offers else offers
        if isinstance(offer, dict):
            page.price = page.price or _as_float(offer.get("price") or offer.get("lowPrice"))
            page.currency = page.currency or str(offer.get("priceCurrency") or "")
            page.availability = page.availability or str(offer.get("availability") or "").split("/")[-1]
            for key in ("sku", "mpn"):
                if offer.get(key):
                    page.raw_identifiers.setdefault(f"offer_{key}", str(offer[key]))
        page.images.extend(_collect_image_urls({"image": node.get("image")}, base))
        color = node.get("color")
        if color and "color" not in page.raw_identifiers:
            page.raw_identifiers["color"] = str(color)


# ── Shopify ──

def _shopify_json(client: httpx.Client, url: str) -> Optional[dict]:
    m = re.search(r"^(https?://[^/]+)(?:/[^?#]*)?/products/([A-Za-z0-9._%-]+)", url)
    if not m:
        return None
    resp = _get(client, f"{m.group(1)}/products/{m.group(2)}.json", accept_json=True)
    if resp is None:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data.get("product") if isinstance(data, dict) else None


def _apply_shopify(page: ProductPageData, product: dict, base: str, brand_hint: Optional[str]) -> None:
    page.extraction_sources.append("shopify_json")
    page.title = page.title or str(product.get("title") or "")
    page.brand = page.brand or str(product.get("vendor") or "")
    page.description = page.description or re.sub(r"<[^>]+>", " ", str(product.get("body_html") or ""))
    page.category = page.category or str(product.get("product_type") or "")
    for img in product.get("images") or []:
        if isinstance(img, dict) and img.get("src"):
            page.images.append(urljoin(base, img["src"]))
    variants = product.get("variants") or []
    for v in variants:
        if not isinstance(v, dict):
            continue
        if v.get("sku") and "variant_sku" not in page.raw_identifiers:
            page.raw_identifiers["variant_sku"] = str(v["sku"])
        if v.get("barcode") and "barcode" not in page.raw_identifiers:
            page.raw_identifiers["barcode"] = str(v["barcode"])
        if page.price is None and v.get("price"):
            page.price = _as_float(v["price"])
        for key in ("option1", "option2", "option3", "title"):
            val = v.get(key)
            if isinstance(val, str) and re.search(r"\d", val) and len(val) <= 12 and val not in page.sizes:
                page.sizes.append(val)
                break
    tags = product.get("tags")
    if isinstance(tags, list):
        page.breadcrumbs = page.breadcrumbs or [str(t) for t in tags[:15]]


# ── Embedded app state ──

def _embedded_json_blobs(html: str) -> List[Any]:
    blobs = []
    for m in re.finditer(r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', html, re.DOTALL):
        try:
            blobs.append(json.loads(m.group(1)))
        except ValueError:
            pass
    for m in re.finditer(r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>', html, re.DOTALL):
        try:
            blobs.append(json.loads(m.group(1)))
        except ValueError:
            pass
    for m in re.finditer(r'window\.__(?:INITIAL_STATE|PRELOADED_STATE|APOLLO_STATE|NUXT|STATE)__\s*=\s*(\{.*?\})\s*;?\s*</script>',
                         html, re.DOTALL):
        try:
            blobs.append(json.loads(m.group(1)))
        except ValueError:
            pass
    return blobs


# ── HTML ──

def _apply_html(page: ProductPageData, html: str, base: str) -> str:
    tree = HTMLParser(html)
    page.extraction_sources.append("html")

    def meta(*names) -> str:
        for name in names:
            for attr in ("property", "name", "itemprop"):
                node = tree.css_first(f'meta[{attr}="{name}"]')
                if node and node.attributes.get("content"):
                    return node.attributes["content"].strip()
        return ""

    og_title = meta("og:title", "twitter:title")
    if og_title:
        page.extraction_sources.append("opengraph")
    page.title = page.title or og_title or (tree.css_first("title").text(strip=True) if tree.css_first("title") else "")
    h1 = tree.css_first("h1")
    if h1 and (not page.title or len(page.title) > 120):
        page.title = h1.text(strip=True) or page.title
    page.description = page.description or meta("og:description", "description")
    page.brand = page.brand or meta("product:brand", "og:brand", "brand")
    page.price = page.price or _as_float(meta("product:price:amount", "og:price:amount", "price"))
    page.currency = page.currency or meta("product:price:currency", "og:price:currency")
    for key, name in (("sku", "product:retailer_item_id"), ("mpn", "product:mfr_part_no"), ("sku2", "sku")):
        val = meta(name)
        if val:
            page.raw_identifiers.setdefault(key, val)

    for name in ("og:image", "og:image:secure_url", "twitter:image"):
        for node in tree.css(f'meta[property="{name}"], meta[name="{name}"]'):
            if node.attributes.get("content"):
                page.images.append(urljoin(base, node.attributes["content"]))

    if not page.breadcrumbs:
        for sel in ('nav[aria-label*="readcrumb"] a', ".breadcrumb a", ".breadcrumbs a", 'ol[class*="readcrumb"] a',
                    '[itemtype*="BreadcrumbList"] a'):
            names = [a.text(strip=True) for a in tree.css(sel) if a.text(strip=True)]
            if len(names) >= 2:
                page.breadcrumbs = names
                break

    img_urls = []
    for img in tree.css("img"):
        src = img.attributes.get("src") or img.attributes.get("data-src") or img.attributes.get("data-zoom-image") or ""
        srcset = img.attributes.get("srcset") or img.attributes.get("data-srcset") or ""
        if srcset:
            best = srcset.split(",")[-1].strip().split(" ")[0]
            src = best or src
        if not src or _IMG_SKIP_RE.search(src) or not _IMG_EXT_RE.search(src):
            continue
        alt = (img.attributes.get("alt") or "").lower()
        w = img.attributes.get("width")
        if w and w.isdigit() and int(w) < 150:
            continue
        score = 0
        if page.title and any(tok in alt for tok in page.title.lower().split()[:3]):
            score += 2
        if re.search(r"product|gallery|zoom|pdp|main|hero|detail", (img.attributes.get("class") or "") + src, re.IGNORECASE):
            score += 1
        img_urls.append((score, urljoin(base, src)))
    img_urls.sort(key=lambda x: -x[0])
    page.images.extend(u for _, u in img_urls)

    for node in tree.css("script, style, noscript, svg, header, footer, nav"):
        node.decompose()
    body = tree.body
    text = body.text(separator=" ", strip=True) if body else ""
    return re.sub(r"\s+", " ", text)[:60000]


# ── Orchestration ──

def extract_product_page(url: str, client: Optional[httpx.Client] = None) -> ProductPageData:
    page = ProductPageData(url=url)
    try:
        _, host = _assert_public_url(url)
    except FetchBlocked as exc:
        page.error = f"blocked:{exc}"
        return page
    page.domain = host.lower().removeprefix("www.")

    own = client is None
    client = client or httpx.Client()
    try:
        resp = _get(client, url)
        if resp is None:
            page.error = "fetch_failed"
            return page
        html = resp.text
        final_url = str(resp.url)
        page.final_url = final_url
        shopify = _shopify_json(client, url) if "/products/" in url else None
        if shopify is None and "/products/" in final_url and final_url != url:
            shopify = _shopify_json(client, final_url)
        return extract_from_html(html, final_url, page=page, shopify_product=shopify)
    finally:
        if own:
            client.close()


def extract_from_html(html: str, url: str, page: Optional[ProductPageData] = None,
                      shopify_product: Optional[dict] = None) -> ProductPageData:
    """Pure function over already-fetched content — used by tests and by
    callers that already have the HTML."""
    page = page or ProductPageData(url=url, domain=urlparse(url).hostname or "")
    base = url
    brand_hint = None

    nodes = _jsonld_products(html)
    if nodes:
        _apply_jsonld(page, nodes, base)
    if shopify_product:
        _apply_shopify(page, shopify_product, base, brand_hint)
    blobs = _embedded_json_blobs(html)
    if blobs:
        page.extraction_sources.append("embedded_json")
        for blob in blobs:
            for k, v in _collect_identifiers(blob).items():
                page.raw_identifiers.setdefault(f"embedded_{k}", v)
            page.images.extend(_collect_image_urls(blob, base))
    visible_text = _apply_html(page, html, base)

    page.brand = canonical_brand(page.brand) if page.brand else ""
    brand_hint = sku_detector.brand_family_from_text(" ".join([page.brand, page.title, page.category]))
    page.images = _dedupe(page.images)
    page.extraction_sources = list(dict.fromkeys(page.extraction_sources))

    groups: List[List[StyleCodeCandidate]] = []
    ident_cands = []
    for field, value in page.raw_identifiers.items():
        c = sku_detector.candidate_from_identifier(value, field, "page", brand_hint)
        if c:
            ident_cands.append(c)
    groups.append(ident_cands)
    groups.append(sku_detector.detect_style_codes(page.title, "page:title", brand_hint, base_confidence=0.6))
    # Retailers often put the style code in the URL slug (/products/...-ir2175-400).
    # Use the URL that was asked for, not where it redirected to.
    slug = urlparse(page.url or url).path.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")
    groups.append(sku_detector.detect_style_codes(slug, "page:url", brand_hint, base_confidence=0.5))
    groups.append(sku_detector.detect_style_codes(page.description, "page:description", brand_hint, base_confidence=0.55))
    groups.append(sku_detector.detect_style_codes(visible_text, "page:body", brand_hint, base_confidence=0.45))
    page.sku_candidates = sku_detector.merge_candidates(groups)
    if not page.title and not page.images:
        page.error = page.error or "no_metadata"
    return page
