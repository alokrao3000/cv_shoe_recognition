"""
Canonical sneaker representation: brand / model / sub-model / colorway /
gender / size category, from messy retailer or LLM text. Keeps the original
text alongside (callers store both).
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

# ── Brands ──

_BRAND_ALIASES: List[Tuple[str, Tuple[str, ...]]] = [
    ("Jordan", ("air jordan", "jordan brand", "jordan")),
    ("Nike", ("nike sb", "nike", "nikelab", "nike acg")),
    ("adidas", ("adidas originals", "adidas", "yeezy", "y-3")),
    ("New Balance", ("new balance", "nb")),
    ("ASICS", ("asics", "asics sportstyle")),
    ("Converse", ("converse",)),
    ("Vans", ("vans", "vault by vans")),
    ("Puma", ("puma",)),
    ("Reebok", ("reebok",)),
    ("Salomon", ("salomon",)),
    ("On", ("on running", "on cloud", "on ")),
    ("Hoka", ("hoka one one", "hoka")),
    ("Under Armour", ("under armour", "ua curry", "curry brand")),
    ("Saucony", ("saucony",)),
    ("Timberland", ("timberland",)),
    ("UGG", ("ugg",)),
    ("Crocs", ("crocs",)),
    ("Birkenstock", ("birkenstock",)),
    ("Dr. Martens", ("dr. martens", "dr martens", "doc martens")),
    ("Balenciaga", ("balenciaga",)),
    ("Gucci", ("gucci",)),
    ("Louis Vuitton", ("louis vuitton",)),
    ("Bape", ("a bathing ape", "bape")),
    ("Golden Goose", ("golden goose",)),
    ("Common Projects", ("common projects",)),
    ("Fear of God", ("fear of god", "essentials")),
    ("Maison Margiela", ("maison margiela", "margiela", "mm6")),
]


def known_brand(text: str) -> str:
    """Canonical brand if the text mentions one we know, else ''."""
    low = f" {(text or '').lower()} "
    for canon, aliases in _BRAND_ALIASES:
        for a in aliases:
            if f" {a}" in low or low.strip() == a:
                return canon
    return ""


def canonical_brand(text: str) -> str:
    """For explicit brand fields: canonicalize known brands, keep unknown ones as given."""
    return known_brand(text) or (text or "").strip()


def brand_family(brand: str) -> str:
    """Nike and Jordan share a catalog/style-code system; treat as one family."""
    b = canonical_brand(brand)
    return "Nike" if b in ("Nike", "Jordan") else b


# ── Models ──
# canonical model → regex alternatives (matched case-insensitively against a
# title with the brand removed). Longer/more specific patterns first.

# "<model> ... low" with up to a few intervening words ("Jordan 1 Little Kids Retro Low"),
# but no other height word in between.
def _height(base: str, height: str, others: str) -> str:
    return rf"{base}\b(?:(?!\b(?:{others})\b)[\w'’\"()/-]*\s*){{0,6}}?\b{height}\b"


_J1 = r"(?:air\s*)?jordan\s*1|aj\s*1"
_AF1 = r"air\s*force\s*1|\baf1"
_MODEL_PATTERNS: List[Tuple[str, str]] = [
    # Jordan
    ("Air Jordan 1 Low", _height(f"(?:{_J1})", "low", "high|hi|mid")),
    ("Air Jordan 1 Mid", _height(f"(?:{_J1})", "mid", "high|hi|low")),
    ("Air Jordan 1 High", _height(f"(?:{_J1})", "(?:high|hi)", "low|mid") + rf"|(?:{_J1})\b(?:(?!\b(?:low|mid|high|hi)\b)[\w'’\"()/-]*\s*){{0,8}}$"),
    ("Air Jordan 2", r"(?:air\s*)?jordan\s*2\b|aj\s*2\b"),
    ("Air Jordan 3", r"(?:air\s*)?jordan\s*3\b|aj\s*3\b"),
    ("Air Jordan 4", r"(?:air\s*)?jordan\s*4\b|aj\s*4\b"),
    ("Air Jordan 5", r"(?:air\s*)?jordan\s*5\b|aj\s*5\b"),
    ("Air Jordan 6", r"(?:air\s*)?jordan\s*6\b|aj\s*6\b"),
    ("Air Jordan 7", r"(?:air\s*)?jordan\s*7\b|aj\s*7\b"),
    ("Air Jordan 8", r"(?:air\s*)?jordan\s*8\b|aj\s*8\b"),
    ("Air Jordan 9", r"(?:air\s*)?jordan\s*9\b|aj\s*9\b"),
    ("Air Jordan 10", r"(?:air\s*)?jordan\s*10\b|aj\s*10\b"),
    ("Air Jordan 11 Low", _height(r"(?:(?:air\s*)?jordan\s*11|aj\s*11)", "low", "high|hi|mid")),
    ("Air Jordan 11", r"(?:air\s*)?jordan\s*11\b|aj\s*11\b"),
    ("Air Jordan 12", r"(?:air\s*)?jordan\s*12\b|aj\s*12\b"),
    ("Air Jordan 13", r"(?:air\s*)?jordan\s*13\b|aj\s*13\b"),
    ("Air Jordan 14", r"(?:air\s*)?jordan\s*14\b|aj\s*14\b"),
    # Nike
    ("Nike SB Dunk Low", r"sb\s*dunk\s*low|dunk\s*low\s*(?:pro\s*)?sb"),
    ("Nike SB Dunk High", r"sb\s*dunk\s*(?:high|hi)|dunk\s*(?:high|hi)\s*(?:pro\s*)?sb"),
    ("Nike Dunk Low", _height(r"\bdunk", "low", "high|hi|mid")),
    ("Nike Dunk High", _height(r"\bdunk", "(?:high|hi)", "low|mid")),
    ("Nike Dunk", r"\bdunk\b"),
    ("Nike Air Force 1 Mid", _height(f"(?:{_AF1})", "mid", "high|hi|low")),
    ("Nike Air Force 1 High", _height(f"(?:{_AF1})", "(?:high|hi)", "low|mid")),
    ("Nike Air Force 1 Low", rf"(?:{_AF1})\b"),
    ("Nike Air Max 1", r"air\s*max\s*1\b|\bam1\b"),
    ("Nike Air Max 90", r"air\s*max\s*90|\bam90\b"),
    ("Nike Air Max 95", r"air\s*max\s*95"),
    ("Nike Air Max 97", r"air\s*max\s*97"),
    ("Nike Air Max 98", r"air\s*max\s*98"),
    ("Nike Air Max Plus", r"air\s*max\s*plus|\btn\b"),
    ("Nike Air Max 270", r"air\s*max\s*270"),
    ("Nike Air Max 720", r"air\s*max\s*720"),
    ("Nike Air Max DN", r"air\s*max\s*dn"),
    ("Nike Air Max Scorpion", r"air\s*max\s*scorpion"),
    ("Nike Vomero 5", r"vomero\s*5"),
    ("Nike P-6000", r"p-?6000"),
    ("Nike Cortez", r"cortez"),
    ("Nike Blazer Mid", r"blazer\s*mid"),
    ("Nike Blazer Low", r"blazer\s*low"),
    ("Nike Vapormax", r"vapor\s*max"),
    ("Nike Shox", r"\bshox\b"),
    ("Nike Killshot 2", r"killshot"),
    ("Nike Field General", r"field\s*general"),
    ("Nike Kobe", r"\bkobe\b"),
    ("Nike LeBron", r"\blebron\b"),
    ("Nike KD", r"\bkd\s*\d+"),
    ("Nike Zoom Freak", r"zoom\s*freak"),
    ("Nike Ja", r"\bja\s*\d\b"),
    ("Nike Pegasus", r"pegasus"),
    ("Nike Air Huarache", r"huarache"),
    ("Nike Air Presto", r"presto"),
    # adidas
    ("adidas Yeezy Boost 350 V2", r"yeezy\s*(?:boost\s*)?350\s*v2"),
    ("adidas Yeezy Boost 350", r"yeezy\s*(?:boost\s*)?350"),
    ("adidas Yeezy 500", r"yeezy\s*500"),
    ("adidas Yeezy Boost 700", r"yeezy\s*(?:boost\s*)?700"),
    ("adidas Yeezy 380", r"yeezy\s*380"),
    ("adidas Yeezy 450", r"yeezy\s*450"),
    ("adidas Yeezy Slide", r"yeezy\s*slide"),
    ("adidas Yeezy Foam RNNR", r"yeezy\s*foam\s*(?:runner|rnnr)"),
    ("adidas Samba OG", r"samba\s*og"),
    ("adidas Samba", r"\bsamba\b"),
    ("adidas Gazelle Indoor", r"gazelle\s*indoor"),
    ("adidas Gazelle", r"gazelle"),
    ("adidas Handball Spezial", r"handball\s*spezial|spezial"),
    ("adidas Superstar", r"superstar"),
    ("adidas Stan Smith", r"stan\s*smith"),
    ("adidas Campus 00s", r"campus\s*00s"),
    ("adidas Campus", r"\bcampus\b"),
    ("adidas Ultraboost", r"ultra\s*boost"),
    ("adidas NMD", r"\bnmd\b"),
    ("adidas Forum Low", r"forum\s*low"),
    ("adidas Forum High", r"forum\s*(?:high|hi|mid)"),
    ("adidas SL 72", r"\bsl\s*72\b"),
    ("adidas Adizero", r"adizero"),
    # New Balance
    ("New Balance 550", r"\b(?:bb)?550\b"),
    ("New Balance 574", r"\b(?:ml|u)?574\b"),
    ("New Balance 990v6", r"990\s*v6"),
    ("New Balance 990v5", r"990\s*v5"),
    ("New Balance 990v4", r"990\s*v4"),
    ("New Balance 990v3", r"990\s*v3"),
    ("New Balance 990", r"\b(?:m)?990\b"),
    ("New Balance 991", r"\b991\b"),
    ("New Balance 992", r"\b992\b"),
    ("New Balance 993", r"\b993\b"),
    ("New Balance 996", r"\b996\b"),
    ("New Balance 997", r"\b997\b"),
    ("New Balance 998", r"\b998\b"),
    ("New Balance 1906R", r"1906\s*r"),
    ("New Balance 2002R", r"2002\s*r"),
    ("New Balance 9060", r"\b9060\b"),
    ("New Balance 327", r"\b327\b"),
    ("New Balance 530", r"\b530\b"),
    ("New Balance 1000", r"\b1000\b"),
    ("New Balance 480", r"\b480\b"),
    ("New Balance 740", r"\b740\b"),
    # ASICS
    ("ASICS Gel-Kayano 14", r"gel[\s-]*kayano\s*14"),
    ("ASICS Gel-1130", r"gel[\s-]*1130"),
    ("ASICS Gel-NYC", r"gel[\s-]*nyc"),
    ("ASICS Gel-Lyte III", r"gel[\s-]*lyte\s*(?:iii|3)"),
    ("ASICS Gel-1090", r"gel[\s-]*1090"),
    ("ASICS GT-2160", r"gt[\s-]*2160"),
    ("ASICS Gel-Nimbus 9", r"gel[\s-]*nimbus\s*9"),
    ("ASICS Gel-Quantum", r"gel[\s-]*quantum"),
    # Others
    ("Converse Chuck 70", r"chuck\s*(?:taylor\s*)?70"),
    ("Converse Chuck Taylor All Star", r"chuck\s*taylor|all\s*star"),
    ("Converse One Star", r"one\s*star"),
    ("Vans Old Skool", r"old\s*skool"),
    ("Vans Sk8-Hi", r"sk8[\s-]*hi"),
    ("Vans Authentic", r"authentic"),
    ("Vans Knu Skool", r"knu\s*skool"),
    ("Puma Suede", r"\bsuede\b"),
    ("Puma Palermo", r"palermo"),
    ("Puma Speedcat", r"speedcat"),
    ("Puma RS-X", r"rs[\s-]*x"),
    ("Reebok Club C", r"club\s*c\b"),
    ("Reebok Classic Leather", r"classic\s*leather"),
    ("Reebok Question", r"\bquestion\b"),
    ("Salomon XT-6", r"xt[\s-]*6\b"),
    ("Salomon XT-4", r"xt[\s-]*4\b"),
    ("Salomon ACS Pro", r"acs\s*pro"),
    ("Salomon Speedcross", r"speedcross"),
    ("On Cloudmonster", r"cloudmonster"),
    ("On Cloud 5", r"cloud\s*5\b"),
    ("On Cloudswift", r"cloudswift"),
    ("Hoka Bondi", r"bondi"),
    ("Hoka Clifton", r"clifton"),
    ("Under Armour Curry", r"\bcurry\b"),
    ("Crocs Classic Clog", r"classic\s*clog|\bclog\b"),
    ("Timberland 6 Inch Boot", r"6[\s-]*inch|6\"\s*premium"),
    ("UGG Tasman", r"tasman"),
    ("UGG Classic Mini", r"classic\s*mini"),
    ("Birkenstock Boston", r"\bboston\b"),
    ("Birkenstock Arizona", r"arizona"),
]
_MODEL_COMPILED = [(canon, re.compile(pat, re.IGNORECASE)) for canon, pat in _MODEL_PATTERNS]

_SUB_MODEL_TOKENS = ("retro", "og", "lv8", "'07", "premium", "prm", "se", "sp", "qs", "essential",
                     "next nature", "nn", "cmft", "lx", "pro", "b", "flyease", "craft", "utility", "gore-tex",
                     "gtx", "made in usa", "made in uk", "miusa", "miuk")


def canonical_model(text: str) -> Tuple[str, str]:
    """Returns (canonical_model, sub_model_text). Empty model when nothing matched."""
    t = text or ""
    for canon, rx in _MODEL_COMPILED:
        if rx.search(t):
            low = t.lower()
            subs = [tok for tok in _SUB_MODEL_TOKENS if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", low)]
            return canon, " ".join(dict.fromkeys(subs))
    return "", ""


# ── Gender / size category ──

_SIZE_CATEGORY_PATTERNS = [
    ("td", r"\(td\)|\btd\b|toddlers?|\binfants?\b|\bbaby\b|crib"),
    ("ps", r"\(ps\)|\bps\b|pre-?school|little\s*kids?"),
    ("gs", r"\(gs\)|\bgs\b|grade\s*school|big\s*kids?|\byouth\b|\bkids'?\b|\bjunior\b|\bjr\b"),
]
_GENDER_PATTERNS = [
    ("women", r"\(w\)|\bwmns\b|\bwomen'?s?\b|\bwoman'?s?\b|\bwmn\b|\bladies\b|\bfemale\b|\(women'?s\)"),
    ("men", r"\bmen'?s?\b|\bmens\b|\bmale\b"),
]


def parse_gender_and_category(text: str) -> Tuple[str, str]:
    """(gender: men|women|unisex|kids|unknown, size_category: adult|gs|ps|td|unknown)."""
    low = (text or "").lower()
    for cat, pat in _SIZE_CATEGORY_PATTERNS:
        if re.search(pat, low):
            return "kids", cat
    for gender, pat in _GENDER_PATTERNS:
        if re.search(pat, low):
            return gender, "adult"
    return "unknown", "unknown"


def category_compatible(a: str, b: str) -> Optional[bool]:
    """None when either side is unknown; True/False otherwise."""
    if not a or not b or a == "unknown" or b == "unknown":
        return None
    return a == b


def gender_compatible(a: str, b: str) -> Optional[bool]:
    if not a or not b or a in ("unknown", "unisex") or b in ("unknown", "unisex"):
        return None
    return a == b


# ── Colorways ──

_COLOR_SYNONYMS: Dict[str, str] = {
    "wht": "white", "blk": "black", "gry": "grey", "gray": "grey", "univ red": "university red",
    "university red": "university red", "varsity red": "varsity red", "gym red": "gym red",
    "sail": "sail", "cream": "cream", "off white": "off white", "offwhite": "off white", "off-white": "off white",
    "midnight navy": "midnight navy", "navy": "navy", "royal": "royal", "game royal": "game royal",
    "photon dust": "photon dust", "wolf grey": "wolf grey", "cool grey": "cool grey", "smoke grey": "smoke grey",
    "lt": "light", "dk": "dark", "pnk": "pink", "grn": "green", "ylw": "yellow", "pur": "purple",
    "org": "orange", "brn": "brown", "gum": "gum", "coconut milk": "coconut milk", "phantom": "phantom",
    "summit white": "summit white", "pale ivory": "pale ivory", "sesame": "sesame", "bone": "bone",
    "cloud white": "cloud white", "core black": "core black", "collegiate green": "collegiate green",
    "sea salt": "sea salt", "moonbeam": "moonbeam", "aluminum": "aluminum", "silver": "silver",
    "metallic silver": "metallic silver", "metallic gold": "metallic gold", "gold": "gold",
}
_COLOR_WORDS = {
    "white", "black", "grey", "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "navy", "royal", "sail", "cream", "gum", "gold", "silver", "tan", "beige", "teal", "burgundy",
    "maroon", "olive", "khaki", "ivory", "bone", "phantom", "sesame", "mint", "lilac", "coral",
    "turquoise", "aqua", "crimson", "scarlet", "bred", "chicago", "panda", "mocha", "obsidian",
    "off-white", "volt", "infrared", "cement", "military", "neutral", "light", "dark", "pale",
    "metallic", "wolf", "cool", "smoke", "university", "varsity", "gym", "midnight", "photon",
    "dust", "summit", "core", "cloud", "collegiate", "sea", "salt", "moonbeam", "aluminum", "coconut",
    "milk", "game", "pine", "malachite", "lucky", "medium", "true", "bright", "sport", "team",
    "anthracite", "iron", "particle", "platinum", "hemp", "wheat", "mushroom", "taupe", "rattan",
    "sand", "desert", "stone", "concrete", "graphite", "carbon", "ash", "chalk", "linen", "oatmeal",
    "cacao", "chocolate", "espresso", "walnut", "hazel", "amber", "rust", "copper", "bronze", "brass",
    "off white",
}
# Release nicknames and qualifiers: compared with each other, never against real colours.
_NICKNAME_WORDS = {
    "panda", "bred", "chicago", "mocha", "reverse", "inverse", "alternate", "alt", "next", "nature", "shadow",
    "royal", "pine", "lucky", "toe", "cement", "military", "black cat", "cactus", "jack", "fragment", "og",
}


_PHRASES = sorted({k: v for k, v in _COLOR_SYNONYMS.items() if " " in k or "-" in k or " " in v}.items(),
                  key=lambda kv: -len(kv[0]))


def colorway_tokens(text: str) -> List[str]:
    """Ordered, deduplicated colour + nickname tokens. 'White/Black-Gum' →
    ['white','black','gum']. Known multi-word names become one token
    ('Off White' → 'off white', 'University Red' → 'university red'), so
    'Off White/Black' ≠ 'White/Black'."""
    low = (text or "").lower().replace("&", "/").replace(" and ", "/")
    low = re.sub(r"[()\[\]']", " ", low)
    for phrase, canon in _PHRASES:
        low = re.sub(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", canon.replace(" ", "_"), low)
    parts = re.split(r"\s*[/,\-–|]\s*", low)
    out: List[str] = []
    for p in parts:
        for w in p.split():
            w = _COLOR_SYNONYMS.get(w, w).replace("_", " ")
            if (w in _COLOR_WORDS or w in _NICKNAME_WORDS or w in _COLOR_SYNONYMS.values()) and w not in out:
                out.append(w)
    return out


def _split_tokens(tokens: List[str]) -> Tuple[List[str], List[str]]:
    colors = [t for t in tokens if t not in _NICKNAME_WORDS]
    nicks = [t for t in tokens if t in _NICKNAME_WORDS]
    return colors, nicks


_PRIMARY_COLORS = {"white", "black", "grey", "red", "blue", "green", "yellow", "orange", "purple", "pink",
                   "brown", "navy", "sail", "cream", "gum", "gold", "silver", "tan", "beige", "teal", "olive",
                   "royal", "burgundy", "maroon", "obsidian", "off-white", "bone", "ivory", "khaki", "phantom"}


def _primary(tokens: List[str]) -> List[str]:
    prim = []
    for t in tokens:
        for w in t.split():
            if w in _PRIMARY_COLORS and w not in prim:
                prim.append(w)
        if t in _PRIMARY_COLORS and t not in prim:
            prim.append(t)
    return prim


def colorway_similarity(a: str, b: str) -> Optional[float]:
    """0..1. None when either side has no recognizable color tokens.
    Order-sensitive on the primary colour: 'White/Black' vs 'Black/White' are
    different colorways (Panda vs Reverse Panda) — same set, different lead."""
    ta, tb = colorway_tokens(a), colorway_tokens(b)
    if not ta or not tb:
        return None
    ca, na = _split_tokens(ta)
    cb, nb = _split_tokens(tb)
    sims = []
    if ca and cb:
        jaccard = len(set(ca) & set(cb)) / len(set(ca) | set(cb))
        pa, pb = _primary(ca), _primary(cb)
        if pa and pb and pa[0] != pb[0]:
            jaccard *= 0.7
        sims.append(jaccard)
    if na and nb:
        sims.append(len(set(na) & set(nb)) / len(set(na) | set(nb)))
    if not sims:
        # one side only has real colours, the other only a nickname — not comparable
        return None
    return max(0.0, min(1.0, min(sims)))


def model_similarity(a: str, b: str) -> Optional[float]:
    """0..1 on canonical model names; None if either side is empty."""
    ca, _ = canonical_model(a)
    cb, _ = canonical_model(b)
    if ca and cb:
        if ca == cb:
            return 1.0
        # "Nike Dunk" (unspecified height) vs "Nike Dunk Low" — partial credit
        if ca in cb or cb in ca:
            return 0.6
        return fuzz.token_set_ratio(ca, cb) / 100.0 * 0.5
    if not a or not b:
        return None
    return fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0


@dataclass
class ParsedTitle:
    original: str
    brand: str = ""
    model: str = ""
    sub_model: str = ""
    colorway_text: str = ""
    gender: str = "unknown"
    size_category: str = "unknown"
    collab: str = ""
    tokens: List[str] = field(default_factory=list)


_COLLAB_RE = re.compile(r"\b(travis scott|off-white|fragment|union|supreme|sacai|stussy|st[üu]ssy|ambush|"
                        r"cactus jack|a ma mani[eé]re|j balvin|dior|tiffany|ben & jerry'?s|strangelove|"
                        r"parra|concepts|bodega|kith|undefeated|atmos|patta|clot|aim[eé] leon dore|ald|"
                        r"jjjjound|joe freshgoods|nigo|wales bonner|bad bunny|pharrell|gucci|prada|"
                        r"louis vuitton|comme des gar[cç]ons|cdg|futura|jarritos|born x raised|"
                        r"salehe bembury|ronnie fieg|sean wotherspoon|tom sachs|hello kitty|powerpuff)\b",
                        re.IGNORECASE)


def parse_title(title: str, brand_hint: str = "") -> ParsedTitle:
    t = (title or "").strip()
    brand = canonical_brand(brand_hint) if brand_hint else known_brand(t)
    if brand and brand == (brand_hint or "").strip() and known_brand(t) == "Jordan":
        brand = "Jordan"
    model, sub = canonical_model(t)
    if model.startswith("Air Jordan"):
        brand = "Jordan"
    elif model.startswith("Nike") and not brand:
        brand = "Nike"
    elif model.startswith("adidas") and not brand:
        brand = "adidas"
    gender, cat = parse_gender_and_category(t)
    collab_m = _COLLAB_RE.search(t)
    collab = collab_m.group(1) if collab_m else ""
    # Colorway text: what's left after brand/model words, in quotes, or after a separator
    color_text = ""
    m = re.search(r"['\"“‘]([^'\"”’]+)['\"”’]", t)
    if m:
        color_text = m.group(1)
    else:
        rest = t
        if model:
            for canon, rx in _MODEL_COMPILED:
                if canon == model:
                    rest = rx.sub(" ", rest)
                    break
        for _, aliases in _BRAND_ALIASES:
            for a in aliases:
                rest = re.sub(rf"(?i)\b{re.escape(a)}\b", " ", rest)
        rest = re.sub(r"(?i)\b(retro|og|low|mid|high|hi|premium|prm|men'?s|women'?s|wmns|gs|ps|td|shoes?|sneakers?|"
                      r"boots?|trainers?|\d{1,2}(\.\d)?|size)\b", " ", rest)
        rest = re.sub(r"[-–|:]+", "/", rest)
        color_text = re.sub(r"\s+", " ", rest).strip(" /")
    return ParsedTitle(original=t, brand=brand, model=model, sub_model=sub, colorway_text=color_text,
                       gender=gender, size_category=cat, collab=collab,
                       tokens=[w for w in re.split(r"\W+", t.lower()) if w])
