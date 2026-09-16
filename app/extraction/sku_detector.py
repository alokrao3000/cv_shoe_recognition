"""
Find, normalize and validate sneaker style codes in free text and structured
fields.

A style code is only accepted when it matches a known brand format — a
number that merely looks code-shaped (a date, a price, a UPC, an order id)
must not become a SKU candidate. Labeled occurrences ("Style: DD1391-100")
outrank bare ones, and a bare Nike-shaped code written without its dash
("DD1391100") is only accepted when the surrounding context says Nike.
"""
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from app.schemas import StyleCodeCandidate

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize_style_code(code: str) -> str:
    """Canonical comparison key: uppercase, A-Z0-9 only.
    DD1391-100 / DD1391100 / dd1391 100 → DD1391100."""
    return _NON_ALNUM.sub("", (code or "").upper())


@dataclass(frozen=True)
class _Format:
    brand: str            # brand family this format belongs to
    pattern: re.Pattern   # matched against a single token (no anchors needed; fullmatch used)
    display: "callable"   # normalized → conventional display form
    bare_ok: bool = True  # may be accepted without a label / brand hint


def _dash_display(prefix_len: int):
    def fmt(norm: str) -> str:
        return f"{norm[:prefix_len]}-{norm[prefix_len:]}"
    return fmt


def _identity(norm: str) -> str:
    return norm


# Order matters: more specific formats first. Patterns are applied to the
# NORMALIZED token (A-Z0-9), so separators are irrelevant here — separator
# handling lives in the tokenizer below.
_FORMATS: List[_Format] = [
    # Nike / Jordan modern: 2 letters + 4 digits + 3 digits (DD1391-100), sometimes a trailing letter
    _Format("nike", re.compile(r"[A-Z]{2}\d{4}\d{3}[A-Z]?"), _dash_display(6)),
    # Nike / Jordan legacy: 6 digits + 3 digits (555088-101)
    _Format("nike", re.compile(r"\d{6}\d{3}"), _dash_display(6)),
    # ASICS: 1201A019-020 / 1203A175.100
    _Format("asics", re.compile(r"\d{4}[A-Z]\d{3}\d{3}"), _dash_display(8)),
    # Vans: VN0A38G1EO2 / VN000D3HY28
    _Format("vans", re.compile(r"VN0[A-Z0-9]{8,9}"), _identity),
    # Timberland: TB010061713
    _Format("timberland", re.compile(r"TB0\d{5}[A-Z0-9]{3}"), _identity),
    # Salomon: L47293900
    _Format("salomon", re.compile(r"L\d{8}"), _identity),
    # Converse: 162050C / A02123C
    _Format("converse", re.compile(r"(?:\d{6}|A\d{5})C"), _identity),
    # New Balance: M990GL6, U9060ECC, BB550WT1, ML2002RA, MR530SG, U574LGVA
    _Format("new balance", re.compile(r"[A-Z]{1,2}\d{3,4}[A-Z]{2,4}\d?"), _identity),
    # Hoka: 1123194-BBLC
    _Format("hoka", re.compile(r"\d{7}[A-Z]{3,4}"), _dash_display(7)),
    # UGG: 1016222-CHE
    _Format("ugg", re.compile(r"\d{7}[A-Z]{3}"), _dash_display(7)),
    # Puma: 374915-01 / 397545_01
    _Format("puma", re.compile(r"\d{6}\d{2}"), _dash_display(6)),
    # Crocs: 10001-001 (5 digits + 3)
    _Format("crocs", re.compile(r"\d{5}\d{3}"), _dash_display(5)),
    # On: 61.98025
    _Format("on", re.compile(r"\d{2}\d{5}"), lambda n: f"{n[:2]}.{n[2:]}"),
    # adidas / Reebok / Yeezy: 2 letters + 4 digits (GX8862, IE1234) or letter + 5 digits (B75806)
    _Format("adidas", re.compile(r"[A-Z]{2}\d{4}|[A-Z]\d{5}"), _identity),
    # Reebok newer 9-digit article numbers (100033793) — only when labeled
    _Format("reebok", re.compile(r"1000\d{5}"), _identity, bare_ok=False),
]

# Tokens that pass a format regex but are common false positives.
_DATE_RE = re.compile(r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])?$")
_YEAR_MONTH_RE = re.compile(r"^(19|20)\d{2}\d{2}$")

# A "raw token" is a run of alphanumerics possibly joined by dash, dot,
# underscore or slash. Spaces never join (they'd swallow neighbouring words);
# the one space-separated form that matters ("DD1391 100") has its own regex.
_TOKEN_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]+(?:[-._/][A-Z0-9]+){0,3})(?![A-Z0-9])", re.IGNORECASE)
_SPACED_NIKE_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2}\d{4}|\d{6})\s+(\d{3})(?![A-Z0-9])", re.IGNORECASE)

_LABEL_RE = re.compile(
    r"(?:style\s*(?:code|number|no\.?|#|id|colou?r)?|sku|item\s*(?:#|no\.?|number)?|product\s*code|"
    r"mpn|manufacturer\s*(?:#|number|part)?|model\s*(?:number|no\.?|#)?|article\s*(?:number|no\.?|#)?|"
    r"art\.?\s*(?:no\.?|#)?|colou?r\s*code|reference|ref\.?)\s*[:#.\-]?\s*"
    r"([A-Z0-9][A-Z0-9\-._/]{3,14}(?:\s\d{3})?)",
    re.IGNORECASE,
)


def _prefixes(token: str) -> List[str]:
    """'DD1391-100-10.5' → ['DD1391-100-10.5', 'DD1391-100-10', 'DD1391-100', 'DD1391'] —
    retailers append size/variant segments; try the longest valid prefix."""
    pieces = re.findall(r"[A-Z0-9]+|[-._/ ]", token, re.IGNORECASE)
    out = []
    for n in range(len(pieces), 0, -1):
        if re.fullmatch(r"[-._/ ]", pieces[n - 1]):
            continue
        out.append("".join(pieces[:n]).strip())
    return out


def _first_valid(token: str, brand_hint: Optional[str], labeled: bool) -> "tuple[str, _Format] | None":
    for prefix in _prefixes(token):
        fmt = match_format(prefix, brand_hint, labeled)
        if fmt:
            return prefix, fmt
    return None

_BRAND_HINTS = {
    "nike": ("nike", "jordan", "air jordan", "nike sb", "dunk", "air force", "air max", "blazer", "cortez"),
    "adidas": ("adidas", "yeezy", "samba", "gazelle", "superstar", "stan smith", "ultraboost", "nmd", "forum",
               "campus", "spezial"),
    "new balance": ("new balance", "nb "),
    "asics": ("asics", "gel-"),
    "vans": ("vans", "old skool", "sk8-hi"),
    "converse": ("converse", "chuck"),
    "puma": ("puma",),
    "salomon": ("salomon",),
    "reebok": ("reebok",),
    "hoka": ("hoka",),
    "on": ("on running", "on cloud", "cloudmonster", "cloudswift"),
    "timberland": ("timberland",),
    "ugg": ("ugg",),
    "crocs": ("crocs",),
}

# Brand families whose bare codes are too generic to trust without a hint.
_HINT_REQUIRED = {"adidas", "puma", "crocs", "on", "hoka", "ugg", "new balance"}
# Families that share a surface form and need the hint to disambiguate.
_AMBIGUOUS_FAMILIES = {"puma": {"crocs", "on"}, "crocs": {"puma", "on"}, "on": {"puma", "crocs"},
                       "hoka": {"ugg"}, "ugg": {"hoka"}}


def brand_family_from_text(text: str) -> Optional[str]:
    low = (text or "").lower()
    for family, hints in _BRAND_HINTS.items():
        if any(h in low for h in hints):
            return family
    return None


def _looks_like_false_positive(norm: str) -> bool:
    if norm.isdigit():
        if _DATE_RE.match(norm) or _YEAR_MONTH_RE.match(norm):
            return True
        if len(norm) >= 12:          # UPC/EAN/GTIN — captured separately, never a style code
            return True
    return False


def match_format(token: str, brand_hint: Optional[str] = None,
                 labeled: bool = False) -> Optional[_Format]:
    """Returns the first format the token satisfies, honoring brand hints and
    the labeled/bare rules. None when the token isn't a plausible style code."""
    norm = normalize_style_code(token)
    if len(norm) < 6 or len(norm) > 13 or _looks_like_false_positive(norm):
        return None
    had_separator = bool(re.search(r"[-. _/]", token.strip()))
    matches = [f for f in _FORMATS if f.pattern.fullmatch(norm)]
    if not matches:
        return None
    if brand_hint:
        preferred = [f for f in matches if f.brand == brand_hint]
        if preferred:
            return preferred[0]
        # A token that only fits *another* brand's format than the context's
        # brand is a false positive even when labeled — on a Nike page an
        # adidas-shaped "CD0461" is a truncated Nike prefix, not a SKU.
        return None
    for f in matches:
        if not f.bare_ok and not labeled:
            continue
        if f.brand == "nike" and not had_separator and not labeled and norm[:2].isalpha():
            # DD1391100 without a dash and without a Nike hint: too risky.
            continue
        if f.brand in _HINT_REQUIRED and not labeled:
            continue
        return f
    return None


def _make_candidate(token: str, fmt: _Format, source: str, labeled: bool,
                    brand_hint: Optional[str], confidence: float) -> StyleCodeCandidate:
    norm = normalize_style_code(token)
    return StyleCodeCandidate(code=norm, display=fmt.display(norm), source=source,
                              labeled=labeled, brand_hint=brand_hint or fmt.brand,
                              confidence=confidence)


def detect_style_codes(text: str, source: str, brand_hint: Optional[str] = None,
                       base_confidence: float = 0.5, max_candidates: int = 10) -> List[StyleCodeCandidate]:
    """Scans free text. Labeled hits first (higher confidence), then bare
    tokens. Deduplicated by normalized code, best confidence wins."""
    if not text:
        return []
    brand_hint = brand_hint or brand_family_from_text(text)
    found: dict = {}

    for m in _LABEL_RE.finditer(text):
        hit = _first_valid(m.group(1).strip(" .,;:"), brand_hint, labeled=True)
        if hit:
            cand = _make_candidate(hit[0], hit[1], source, True, brand_hint, min(0.95, base_confidence + 0.4))
            if cand.code not in found or found[cand.code].confidence < cand.confidence:
                found[cand.code] = cand

    bare_tokens = [m.group(1) for m in _TOKEN_RE.finditer(text)]
    bare_tokens += [f"{m.group(1)}-{m.group(2)}" for m in _SPACED_NIKE_RE.finditer(text)]
    for token in bare_tokens:
        if len(found) >= max_candidates:
            break
        hit = _first_valid(token, brand_hint, labeled=False)
        if hit:
            norm = normalize_style_code(hit[0])
            if norm in found:
                continue
            found[norm] = _make_candidate(hit[0], hit[1], source, False, brand_hint, base_confidence)

    return sorted(found.values(), key=lambda c: -c.confidence)


def candidate_from_identifier(value: str, field: str, source: str,
                              brand_hint: Optional[str] = None) -> Optional[StyleCodeCandidate]:
    """For structured fields (JSON-LD sku/mpn, Shopify variant sku). Strips a
    trailing size segment ("DD1391-100-10.5") the way retailers append it."""
    if not value:
        return None
    hit = _first_valid(str(value).strip(), brand_hint, labeled=True)
    if not hit:
        return None
    return _make_candidate(hit[0], hit[1], f"{source}:{field}", True, brand_hint, 0.9)


def merge_candidates(groups: Iterable[List[StyleCodeCandidate]]) -> List[StyleCodeCandidate]:
    best: dict = {}
    for group in groups:
        for c in group:
            cur = best.get(c.code)
            if cur is None or c.confidence > cur.confidence:
                best[c.code] = c
    return sorted(best.values(), key=lambda c: -c.confidence)


def codes_equivalent(a: str, b: str) -> bool:
    return bool(a) and normalize_style_code(a) == normalize_style_code(b)
