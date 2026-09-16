"""
The identification orchestrator. One RunContext flows through the stages;
every stage records what it saw in the evidence graph, its failure codes
and its timing, so a wrong answer can always be traced to the signal that
produced it.

Persistence is optional: `run_inline()` executes without a DB row (CLI,
eval, synchronous API); `run(ident_id)` executes against a stored
Identification, updating it after every stage so a UI can poll progress.
"""
import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image

from app.config import settings
from app.db import repo
from app.db.models import utcnow
from app.extraction import page_extractor, sku_detector
from app.extraction.normalize import parse_title
from app.pipeline import cache as rcache
from app.pipeline import scoring
from app.pipeline.candidates import CandidatePool
from app.schemas import (
    CanonicalProduct, EvidenceSource, IdentifyResult, ProductPageData, ScoredCandidate, Stage, StockXMatch,
    StockXProduct, VerificationResult, VisionAnalysis,
)
from app.search import provider as search_provider
from app.search import web_discovery
from app.stockx.catalog import StockXCatalog, choose_product
from app.stockx.client import StockXBudgetExhausted, StockXRequestFailed
from app.vision.claude_vision import VisionClient, VisionError

logger = logging.getLogger(__name__)

STAGES = [
    ("ingest", "Inputs received"),
    ("extract", "Product page extracted"),
    ("sku", "Style code detection"),
    ("vision", "Image analyzed"),
    ("embedding", "Visual similarity search"),
    ("candidates", "Candidate products found"),
    ("resolve", "Evidence scored"),
    ("verify", "Match verified"),
    ("stockx", "StockX product matched"),
    ("market", "Market data retrieved"),
]


@dataclass
class RunInput:
    images: List[bytes] = field(default_factory=list)
    image_urls: List[str] = field(default_factory=list)
    product_url: str = ""
    title: str = ""
    description: str = ""


@dataclass
class RunContext:
    id: str
    inp: RunInput
    images: List[bytes] = field(default_factory=list)
    image_hashes: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    page: Optional[ProductPageData] = None
    vision: Optional[VisionAnalysis] = None
    query_vectors: List[np.ndarray] = field(default_factory=list)
    pool: CandidatePool = field(default_factory=CandidatePool)
    obs: Optional[scoring.Observation] = None
    verification: Optional[VerificationResult] = None
    stockx_product: Optional[StockXProduct] = None
    stockx_confirmed: bool = False
    stages: List[Stage] = field(default_factory=lambda: [Stage(name=n, label=l) for n, l in STAGES])
    evidence: Dict = field(default_factory=dict)
    failure_codes: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    reference_images: Dict[str, bytes] = field(default_factory=dict)
    result: Optional[IdentifyResult] = None
    on_update: Optional[Callable[["RunContext"], None]] = None

    def fail(self, code: str, detail: str = "") -> None:
        if code not in self.failure_codes:
            self.failure_codes.append(code)
        if detail:
            self.evidence.setdefault("errors", {}).setdefault(code, []).append(detail[:300])

    def method(self, name: str) -> None:
        if name not in self.methods:
            self.methods.append(name)

    def stage(self, name: str) -> Stage:
        return next(s for s in self.stages if s.name == name)

    def notify(self) -> None:
        if self.on_update:
            try:
                self.on_update(self)
            except Exception:
                logger.exception("stage update callback failed")


class Pipeline:
    def __init__(self, catalog: Optional[StockXCatalog] = None, provider=None, vision: Optional[VisionClient] = None,
                 embed_fn: Optional[Callable[[List[bytes]], List[np.ndarray]]] = None, db_enabled: bool = True):
        self.catalog = catalog or StockXCatalog(use_db_cache=db_enabled)
        self.provider = provider or search_provider.get_provider()
        self.vision = vision or VisionClient()
        self._embed_fn = embed_fn
        self.db_enabled = db_enabled

    # ── embeddings (lazy import keeps torch out of test/startup paths) ──

    def embed(self, images: List[bytes]) -> List[np.ndarray]:
        if self._embed_fn is not None:
            return self._embed_fn(images)
        from app.vision import embeddings
        pil = []
        for data in images:
            try:
                pil.append(embeddings.load_image(data))
            except Exception:
                continue
        if not pil:
            return []
        return list(embeddings.embed_images(pil))

    # ── public entry points ──

    def run_inline(self, inp: RunInput, ident_id: Optional[str] = None,
                   on_update: Optional[Callable[[RunContext], None]] = None) -> IdentifyResult:
        ctx = RunContext(id=ident_id or str(uuid.uuid4()), inp=inp, on_update=on_update)
        return self._execute(ctx)

    # ── stage runner ──

    def _timed(self, ctx: RunContext, name: str, fn: Callable[[RunContext], Optional[str]]) -> None:
        st = ctx.stage(name)
        st.status = "running"
        ctx.notify()
        t0 = time.perf_counter()
        try:
            detail = fn(ctx)
            st.status = "done" if detail != "__skip__" else "skipped"
            st.detail = "" if detail in (None, "__skip__") else detail
        except _Skip as exc:
            st.status, st.detail = "skipped", str(exc)
        except Exception as exc:
            logger.exception(f"[{ctx.id}] stage {name} failed")
            st.status, st.detail = "failed", f"{type(exc).__name__}: {exc}"[:300]
        st.ms = int((time.perf_counter() - t0) * 1000)
        ctx.evidence.setdefault("timings_ms", {})[name] = st.ms
        ctx.notify()

    def _execute(self, ctx: RunContext) -> IdentifyResult:
        self._timed(ctx, "ingest", self._ingest)
        cached = self._cache_lookup(ctx)
        if cached is not None:
            ctx.result = cached
            return cached
        self._timed(ctx, "extract", self._extract)
        self._timed(ctx, "sku", self._sku)
        self._timed(ctx, "vision", self._vision)
        if scoring.NOT_A_SNEAKER in ctx.failure_codes:
            for name in ("embedding", "candidates", "resolve", "verify", "stockx", "market"):
                ctx.stage(name).status = "skipped"
            return self._finalize(ctx, scoring.Resolution(None, "unresolved", 0.0, [scoring.NOT_A_SNEAKER], []))
        self._timed(ctx, "embedding", self._embedding)
        self._timed(ctx, "candidates", self._candidates)
        self._timed(ctx, "resolve", self._resolve)
        self._timed(ctx, "verify", self._verify)
        self._timed(ctx, "stockx", self._stockx)
        resolution = scoring.resolve(ctx.pool.all(), ctx.obs, ctx.verification, ctx.stockx_confirmed,
                                     self.vision.available and bool(ctx.images))
        self._timed(ctx, "market", lambda c: self._market(c, resolution))
        return self._finalize(ctx, resolution)

    # ── stages ──

    def _ingest(self, ctx: RunContext) -> str:
        inp = ctx.inp
        images = list(inp.images)
        for url in inp.image_urls[: settings.max_images_per_request]:
            data = page_extractor.fetch_bytes(url)
            if data:
                images.append(data)
            else:
                ctx.fail("IMAGE_DOWNLOAD_FAILED", url)
        ctx.images = _dedupe_images(images)
        ctx.image_hashes = [repo.sha256_bytes(b) for b in ctx.images]
        ctx.evidence["input"] = {
            "url": inp.product_url, "retailer": urlparse(inp.product_url).hostname if inp.product_url else "",
            "title": inp.title, "description": (inp.description or "")[:500],
            "images": len(ctx.images), "image_hashes": ctx.image_hashes,
        }
        if not inp.product_url and not ctx.images and not inp.title:
            raise ValueError("no input: provide an image, an image URL, a product URL or a title")
        return f"{len(ctx.images)} image(s)" + (", product URL" if inp.product_url else "")

    def _cache_lookup(self, ctx: RunContext) -> Optional[IdentifyResult]:
        if not self.db_enabled:
            return None
        key = rcache.cache_key(ctx.image_hashes, ctx.inp.product_url, ctx.inp.title)
        ctx.evidence["input"]["cache_key"] = key
        try:
            from app.db.session import db_session
            with db_session() as s:
                row = rcache.lookup(s, key)
                if row is None or row.id == ctx.id:
                    return None
                result = row_to_result(row)
        except Exception:
            logger.debug("cache lookup failed", exc_info=True)
            return None
        result.id = ctx.id
        if "cache" not in result.identification_method:
            result.identification_method = list(result.identification_method) + ["cache"]
        result.evidence = dict(result.evidence, cache_hit=row.id)
        for st in ctx.stages:
            st.status = "done" if st.name == "ingest" else "skipped"
            if st.name != "ingest":
                st.detail = f"cached from {row.id[:8]}"
        result.stages = ctx.stages
        result.input_images = ctx.image_paths or result.input_images
        return result

    def _extract(self, ctx: RunContext) -> str:
        url = ctx.inp.product_url
        if not url:
            if ctx.inp.title or ctx.inp.description:
                ctx.page = ProductPageData(title=ctx.inp.title or "", description=ctx.inp.description or "")
                ctx.page.sku_candidates = sku_detector.merge_candidates([
                    sku_detector.detect_style_codes(ctx.page.title, "input:title", base_confidence=0.6),
                    sku_detector.detect_style_codes(ctx.page.description, "input:description", base_confidence=0.55),
                ])
                ctx.page.brand = parse_title(ctx.page.title).brand
            raise _Skip("no product URL")
        page = page_extractor.extract_product_page(url)
        structured = bool({"jsonld", "shopify_json", "embedded_json"} & set(page.extraction_sources))
        if page.final_url and urlparse(page.final_url).path.rstrip("/") != urlparse(url).path.rstrip("/"):
            ctx.evidence.setdefault("page_notes", []).append(f"redirected to {page.final_url}")
            if not structured:
                ctx.fail(scoring.NO_METADATA, f"product page redirected to {page.final_url} (product removed?)")
        if ctx.inp.title and (not page.title or not structured):
            # A caller-supplied title beats a title scraped off a page with no
            # structured product data (deleted products redirect to store pages).
            if page.title and page.title != ctx.inp.title:
                ctx.evidence.setdefault("page_notes", []).append(f"page title {page.title!r} replaced by input title")
            page.title = ctx.inp.title
        if ctx.inp.description and not page.description:
            page.description = ctx.inp.description
        ctx.page = page
        ctx.evidence["page"] = {
            "url": page.url, "domain": page.domain, "title": page.title, "brand": page.brand, "price": page.price,
            "currency": page.currency, "category": page.category, "breadcrumbs": page.breadcrumbs[:10],
            "images": page.images[:12], "identifiers": page.raw_identifiers, "sources": page.extraction_sources,
            "sizes": page.sizes[:20], "availability": page.availability, "error": page.error,
        }
        if page.error:
            ctx.fail(scoring.NO_METADATA, page.error)
            return f"failed: {page.error}"
        redirected_away = bool(page.final_url) and urlparse(page.final_url).path.rstrip("/") != urlparse(url).path.rstrip("/")
        if redirected_away and not structured:
            # Whatever images the landing page has are not this product's.
            page.images = []
        if not ctx.images and page.images:
            fetched = []
            with httpx.Client() as client:
                for img_url in page.images[: settings.max_images_per_request]:
                    data = page_extractor.fetch_bytes(img_url, client)
                    if data and _decodable(data):
                        fetched.append(data)
            ctx.images = _dedupe_images(fetched)
            ctx.image_hashes = [repo.sha256_bytes(b) for b in ctx.images]
            ctx.evidence["input"]["images"] = len(ctx.images)
            ctx.evidence["input"]["image_hashes"] = ctx.image_hashes
        if not ctx.images:
            ctx.fail(scoring.NO_PRODUCT_IMAGE)
        if not page.title and not page.images:
            ctx.fail(scoring.NO_METADATA)
        return f"{page.domain}: {len(page.sku_candidates)} code(s), {len(page.images)} image(s) via {','.join(page.extraction_sources)}"

    def _sku(self, ctx: RunContext) -> str:
        codes = list(ctx.page.sku_candidates) if ctx.page else []
        ctx.evidence["sku_candidates"] = [c.model_dump() for c in codes]
        for c in codes:
            ctx.pool.add(c.code, EvidenceSource(kind="page_sku", detail=c.source, weight=c.confidence), display=c.display)
        if codes:
            ctx.method("product_metadata")
            return ", ".join(f"{c.display} ({c.source}{', labeled' if c.labeled else ''})" for c in codes[:4])
        ctx.fail(scoring.NO_SKU)
        return "no style code on the page — falling back to visual + web discovery"

    def _vision(self, ctx: RunContext) -> str:
        if not ctx.images:
            raise _Skip("no images")
        if not self.vision.available:
            ctx.fail(scoring.VISION_UNAVAILABLE)
            raise _Skip("vision model not configured (ANTHROPIC_API_KEY)")
        context = {}
        if ctx.page:
            context = {"title": ctx.page.title, "brand": ctx.page.brand, "description": (ctx.page.description or "")[:600],
                       "detected_style_codes": ", ".join(c.display for c in ctx.page.sku_candidates[:3]),
                       "breadcrumbs": " > ".join(ctx.page.breadcrumbs[:6]), "price": str(ctx.page.price or "")}
        try:
            analysis = self.vision.analyze(ctx.images, context)
        except VisionError as exc:
            ctx.fail(scoring.VISION_UNCERTAIN, f"{exc.code}: {exc.detail}")
            return f"failed: {exc.code}"
        ctx.vision = analysis
        ctx.evidence["vision"] = analysis.model_dump()
        ctx.method("computer_vision")
        if not analysis.is_sneaker:
            ctx.fail(scoring.NOT_A_SNEAKER)
            return "the images do not show a sneaker"
        if not analysis.images_consistent:
            ctx.fail(scoring.CONTRADICTORY_EVIDENCE, "vision: images depict different products")
        if analysis.visible_style_code:
            c = sku_detector.candidate_from_identifier(analysis.visible_style_code, "tag", "vision",
                                                      sku_detector.brand_family_from_text(f"{analysis.brand} {analysis.model}"))
            if c:
                ctx.pool.add(c.code, EvidenceSource(kind="vision_tag", detail="legible on tag/box", weight=0.9), display=c.display)
        for code in analysis.likely_style_codes[:3]:
            c = sku_detector.candidate_from_identifier(code, "guess", "vision",
                                                      sku_detector.brand_family_from_text(f"{analysis.brand} {analysis.model}"))
            if c:
                ctx.pool.add(c.code, EvidenceSource(kind="vision_guess", detail="recalled by vision model", weight=0.35),
                             display=c.display)
        parts = [analysis.brand, analysis.model, analysis.colorway]
        if analysis.official_colorway_guess:
            parts.append(f"'{analysis.official_colorway_guess}'")
        return " ".join(p for p in parts if p) + (f" · tag {analysis.visible_style_code}" if analysis.visible_style_code else "")

    def _embedding(self, ctx: RunContext) -> str:
        if not ctx.images:
            raise _Skip("no images")
        ctx.query_vectors = self.embed(ctx.images)
        if not ctx.query_vectors:
            raise _Skip("images could not be embedded")
        if not self.db_enabled:
            raise _Skip("no database — reference index unavailable")
        from app.db.session import db_session
        from app.db.vector_search import search_similar
        try:
            with db_session() as s:
                hits = search_similar(s, ctx.query_vectors, k=settings.embedding_top_k)
        except Exception as exc:
            ctx.fail("INDEX_UNAVAILABLE", repr(exc))
            raise _Skip(f"reference index unavailable: {type(exc).__name__}")
        ctx.evidence["visual_candidates"] = [
            {"style_code": h.style_code_display, "name": h.name, "similarity": round(h.similarity, 4), "image_url": h.image_url}
            for h in hits
        ]
        for h in hits:
            if h.similarity < settings.embedding_floor:
                continue
            c = ctx.pool.add(h.style_code, EvidenceSource(kind="embedding", detail=f"cosine {h.similarity:.3f}", weight=h.similarity),
                             display=h.style_code_display, name=h.name, brand=h.brand, model=h.model, colorway=h.colorway,
                             gender=h.gender, size_category=h.size_category, image_url=h.image_url, product_db_id=h.product_id)
            if c and (c.embedding_similarity is None or h.similarity > c.embedding_similarity):
                c.embedding_similarity = h.similarity
        if hits:
            ctx.method("image_similarity")
        return f"top match {hits[0].style_code_display} @ {hits[0].similarity:.3f}" if hits else "no reference images indexed yet"

    def _candidates(self, ctx: RunContext) -> str:
        page, vision = ctx.page, ctx.vision
        title = (page.title if page else "") or ctx.inp.title
        brand_hint = sku_detector.brand_family_from_text(" ".join(filter(None, [
            page.brand if page else "", title, vision.brand if vision else "", vision.model if vision else ""])))
        partial = [c.display for c in (page.sku_candidates if page else [])[:2]]
        if vision and vision.visible_style_code:
            partial.insert(0, vision.visible_style_code)

        # Web discovery
        queries = web_discovery.build_queries(page, vision, title, partial_codes=partial)
        web_cands, errors = web_discovery.discover(queries, self.provider, brand_hint, num=settings.web_search_max_results)
        ctx.evidence["web_queries"] = queries
        ctx.evidence["web_candidates"] = [
            {"style_code": w.display, "agreement": w.agreement, "trust": round(w.trust, 2), "domains": w.domains[:6],
             "sources": w.sources[:6], "titles": w.titles[:3]} for w in web_cands[:12]
        ]
        web_by_code = {}
        for w in web_cands[:12]:
            web_by_code[w.code] = w
            ctx.pool.add(w.code, EvidenceSource(kind="web", detail=",".join(w.domains[:4]), weight=min(1.0, w.trust / 2)),
                         display=w.display)
        if web_cands:
            ctx.method("web_search")
        if errors and not web_cands:
            ctx.fail(scoring.WEB_SEARCH_FAILED, "; ".join(errors)[:300])

        # StockX catalog text search — official source of candidate releases
        stockx_queries = []
        if title:
            stockx_queries.append(title)
        if vision and vision.brand and vision.model:
            stockx_queries.append(" ".join(filter(None, [vision.brand, vision.model, vision.official_colorway_guess or vision.colorway])))
            stockx_queries += vision.likely_release_names[:2]
        stockx_hits: List[StockXProduct] = []
        if self.catalog.available:
            seen_q = set()
            for q in stockx_queries[:4]:
                qn = " ".join(q.lower().split())
                if not qn or qn in seen_q:
                    continue
                seen_q.add(qn)
                try:
                    hits = self.catalog.search(q, page_size=10)
                except (StockXRequestFailed, StockXBudgetExhausted) as exc:
                    ctx.fail(scoring.STOCKX_API_FAILED, f"search {q!r}: {exc}")
                    continue
                for h in hits:
                    if h.product_id not in {x.product_id for x in stockx_hits}:
                        stockx_hits.append(h)
        else:
            ctx.fail(scoring.STOCKX_API_FAILED, "not configured")
        ctx.evidence["stockx_candidates"] = [
            {"style_id": h.style_id, "title": h.title, "colorway": h.colorway, "gender": h.gender, "product_id": h.product_id}
            for h in stockx_hits[:20]
        ]
        for h in stockx_hits[:20]:
            ctx.pool.add_stockx_product(h, EvidenceSource(kind="stockx_search", detail="catalog text search", weight=0.2))
        if stockx_hits:
            ctx.method("stockx_catalog")

        # Resolve catalogue metadata for the strongest codes first
        priority = [c.code for c in (page.sku_candidates if page else [])] + partial + [w.code for w in web_cands[:6]]
        errs = ctx.pool.enrich_from_stockx(self.catalog, max_lookups=8, priority=priority)
        if errs and errs != ["stockx_not_configured"]:
            ctx.fail(scoring.STOCKX_API_FAILED, "; ".join(errs)[:300])

        # Our own DB: metadata + reference-image similarity
        if self.db_enabled:
            self._enrich_from_db(ctx)

        ctx.obs = scoring.Observation(page=page, vision=vision, title=title,
                                      page_codes=list(page.sku_candidates) if page else [], web_codes=web_by_code)
        ctx.evidence["detected"] = {
            "brand": ctx.obs.brand, "model": ctx.obs.model, "colorways": ctx.obs.colorway_texts,
            "gender": ctx.obs.gender, "size_category": ctx.obs.size_category, "price": ctx.obs.price,
            "labeled_page_codes": ctx.obs.labeled_page_codes, "tag_code": ctx.obs.visible_tag_code,
        }

        # Reference photos for the strongest image-less candidates, so the visual signal can vote on them
        self._fetch_reference_images(ctx)
        if not ctx.pool.all():
            ctx.fail(scoring.NO_CANDIDATES)
        return f"{len(ctx.pool)} candidate style code(s) from {len(ctx.methods)} signal(s)"

    def _enrich_from_db(self, ctx: RunContext) -> None:
        try:
            from app.db.session import db_session
            from app.db.vector_search import similarity_to_product
            with db_session() as s:
                for c in ctx.pool.all():
                    product = repo.get_product_by_code(s, c.style_code)
                    if product is None:
                        continue
                    c.product_db_id = product.id
                    for key in ("name", "brand", "model", "colorway", "gender", "size_category", "release_date"):
                        if not getattr(c, key) and getattr(product, key):
                            setattr(c, key, getattr(product, key))
                    if c.retail_price is None and product.retail_price:
                        c.retail_price = product.retail_price
                    if not c.stockx_product_id and product.stockx_product_id:
                        c.stockx_product_id = product.stockx_product_id
                        c.stockx_url = f"https://stockx.com/{product.stockx_url_key}" if product.stockx_url_key else ""
                    if not c.image_url and product.images:
                        c.image_url = product.images[0].image_url or ""
                    if ctx.query_vectors and c.embedding_similarity is None:
                        sim = similarity_to_product(s, ctx.query_vectors, product.id)
                        if sim is not None:
                            c.embedding_similarity = sim
                            c.sources.append(EvidenceSource(kind="embedding", detail=f"cosine {sim:.3f} (db)", weight=sim))
        except Exception:
            logger.debug("db enrichment failed", exc_info=True)

    def _fetch_reference_images(self, ctx: RunContext, top_n: int = 5) -> None:
        if not ctx.query_vectors or not self.provider.available:
            return
        prelim = sorted((scoring.score_candidate(c, ctx.obs) for c in ctx.pool.all()), key=lambda c: -c.score)
        targets = [c for c in prelim if c.embedding_similarity is None and (c.name or c.model)][:top_n]
        if not targets:
            return
        fetched = []
        with httpx.Client() as client:
            for c in targets:
                query = f"{c.name or c.model} {c.style_code_display}".strip()
                for img in web_discovery.find_reference_images(query, self.provider, num=6):
                    data = page_extractor.fetch_bytes(img.image_url, client)
                    if not data or not _decodable(data):
                        continue
                    vecs = self.embed([data])
                    if not vecs:
                        continue
                    sim = float(max(np.dot(q, vecs[0]) for q in ctx.query_vectors))
                    c.image_url = c.image_url or img.image_url
                    ctx.reference_images[c.style_code] = data
                    if sim >= settings.embedding_floor:
                        # An unverified web photo can only vote FOR a candidate; a
                        # non-match may just be a bad photo (box, wrong angle, wrong listing).
                        c.embedding_similarity = sim
                        c.sources.append(EvidenceSource(kind="embedding", detail=f"cosine {sim:.3f} (web photo from {img.domain})", weight=sim))
                    else:
                        c.component_notes["embedding_web"] = f"web photo from {img.domain} did not match (cosine {sim:.3f}) — ignored"
                    fetched.append((c, img.image_url, data, vecs[0], sim))
                    break
        ctx.evidence["reference_images"] = [{"style_code": c.style_code_display, "image_url": url, "similarity": round(sim, 4),
                                             "used": sim >= settings.embedding_floor} for c, url, _, _, sim in fetched]
        if fetched and self.db_enabled:
            try:
                from app.db.session import db_session
                with db_session() as s:
                    for c, url, data, vec, sim in fetched:
                        if sim < settings.embedding_floor:
                            continue      # don't index a photo we couldn't even match to the query
                        product = repo.upsert_product(s, c.style_code, name=c.name, brand=c.brand, model=c.model,
                                                      colorway=c.colorway, gender=c.gender, size_category=c.size_category,
                                                      release_date=c.release_date, retail_price=c.retail_price,
                                                      stockx_product_id=c.stockx_product_id,
                                                      stockx_url_key=c.stockx_url.rsplit("/", 1)[-1] if c.stockx_url else "",
                                                      source="web", display=c.style_code_display)
                        repo.add_product_image(s, product, vec, image_url=url, image_bytes=data, source="web")
                        c.product_db_id = product.id
            except Exception:
                logger.debug("storing reference images failed", exc_info=True)

    def _resolve(self, ctx: RunContext) -> str:
        cands = ctx.pool.all()
        for c in cands:
            scoring.score_candidate(c, ctx.obs)
            scoring.detect_contradictions(c, ctx.obs)
        cands.sort(key=lambda c: (c.rejected, -c.score))
        ctx.evidence["scoring"] = {"weights": scoring.weights(),
                                   "ranking": [{"style_code": c.style_code_display, "score": c.score, "rejected": c.rejected,
                                                "reason": c.rejection_reason} for c in cands[:10]]}
        ctx.evidence["contradictions"] = {c.style_code_display: c.contradictions for c in cands if c.contradictions}
        alive = [c for c in cands if not c.rejected]
        if not alive:
            return "every candidate contradicted the evidence" if cands else "no candidates"
        return f"leader {alive[0].style_code_display} @ {alive[0].score:.3f}" + (f", {len(cands) - len(alive)} rejected" if len(cands) > len(alive) else "")

    def _verify(self, ctx: RunContext) -> str:
        alive = sorted([c for c in ctx.pool.all() if not c.rejected], key=lambda c: -c.score)
        top = alive[: settings.verification_top_n]
        if not top:
            raise _Skip("nothing to verify")
        if not ctx.images or not self.vision.available:
            raise _Skip("verifier needs images and a configured vision model")
        refs = dict(ctx.reference_images)
        if self.db_enabled:
            with httpx.Client() as client:
                for c in top:
                    if c.style_code not in refs and c.image_url:
                        data = page_extractor.fetch_bytes(c.image_url, client)
                        if data and _decodable(data):
                            refs[c.style_code] = data
        try:
            ctx.verification = self.vision.verify(ctx.images, top, refs)
        except VisionError as exc:
            ctx.fail(scoring.VISION_UNCERTAIN, f"verify {exc.code}: {exc.detail}")
            return f"failed: {exc.code}"
        scoring.apply_verification(ctx.pool.all(), ctx.verification)
        ctx.evidence["verification"] = ctx.verification.model_dump()
        ctx.method("visual_verification")
        return "; ".join(f"{v.style_code}: {v.verdict} ({v.confidence:.2f})" for v in ctx.verification.verdicts)

    def _stockx(self, ctx: RunContext) -> str:
        alive = sorted([c for c in ctx.pool.all() if not c.rejected], key=lambda c: -c.score)
        if not alive:
            raise _Skip("no surviving candidate")
        winner = alive[0]
        if not self.catalog.available:
            ctx.fail(scoring.STOCKX_API_FAILED, "not configured")
            raise _Skip("StockX not configured")
        try:
            products = self.catalog.by_style_code(winner.style_code_display or winner.style_code)
        except (StockXRequestFailed, StockXBudgetExhausted) as exc:
            ctx.fail(scoring.STOCKX_API_FAILED, str(exc))
            return f"lookup failed: {exc}"
        if not products:
            ctx.fail(scoring.STOCKX_MATCH_UNCERTAIN, f"no StockX product carries {winner.style_code_display}")
            return "no StockX catalog product for the winning style code"
        product, ambiguous = choose_product(products)
        if ambiguous:
            ctx.fail(scoring.STOCKX_MATCH_UNCERTAIN,
                     f"{len(products)} StockX listings share {winner.style_code_display} and disagree: "
                     + " | ".join(p.title for p in products[:4]))
        ctx.stockx_product = product
        CandidatePool.apply_stockx_metadata(winner, product)
        agree = scoring.model_similarity(ctx.obs.model, product.title) if ctx.obs.model else None
        ctx.stockx_confirmed = not ambiguous and (agree is None or agree >= 0.6)
        ctx.method("stockx_match")
        dup = f" ({len(products)} duplicate listings)" if len(products) > 1 and not ambiguous else ""
        return f"{product.title} ({product.style_id}){dup}" + ("" if ctx.stockx_confirmed else " — unconfirmed")

    def _market(self, ctx: RunContext, resolution: scoring.Resolution) -> str:
        if ctx.stockx_product is None or resolution.winner is None:
            raise _Skip("no StockX product")
        if resolution.status == "unresolved":
            raise _Skip("identification unresolved — market data withheld")
        market, err = self.catalog.market(ctx.stockx_product)
        ctx.evidence["market_error"] = err
        if market is None:
            ctx.fail(scoring.STOCKX_API_FAILED, f"market: {err}")
            return f"failed: {err}"
        ctx.evidence["market"] = market.model_dump()
        return f"ask ${market.lowest_ask} / bid ${market.highest_bid}"

    # ── finalize ──

    def _finalize(self, ctx: RunContext, resolution: scoring.Resolution) -> IdentifyResult:
        for code in resolution.failure_codes:
            ctx.fail(code)
        winner = resolution.winner
        status = resolution.status
        confidence = resolution.confidence
        product = None
        stockx = None
        if winner is not None:
            product = CanonicalProduct(
                brand=winner.brand or ctx.obs.brand, model=winner.model or ctx.obs.model, sub_model="",
                colorway=winner.colorway or (ctx.obs.colorway_texts[0] if ctx.obs and ctx.obs.colorway_texts else ""),
                style_code=winner.style_code_display or winner.style_code, gender=winner.gender,
                size_category=winner.size_category, release_date=winner.release_date,
                retailer_url=ctx.inp.product_url, product_name=winner.name, images=[winner.image_url] if winner.image_url else [],
            )
            if ctx.stockx_product is not None:
                market = ctx.evidence.get("market") or {}
                stockx = StockXMatch(
                    product_id=ctx.stockx_product.product_id, url=ctx.stockx_product.url,
                    style_code=ctx.stockx_product.style_id, product_name=ctx.stockx_product.title,
                    match_confidence=confidence if ctx.stockx_confirmed else round(min(confidence, settings.confidence_medium - 0.01), 4),
                    lowest_ask=market.get("lowest_ask"), highest_bid=market.get("highest_bid"),
                    currency=market.get("currency", "USD"), sizes=market.get("sizes", []),
                    market_error=ctx.evidence.get("market_error"),
                )
        ctx.evidence["resolution"] = {
            "selected_style_code": winner.style_code_display if winner else None, "status": status,
            "confidence": confidence, "stockx_product_id": ctx.stockx_product.product_id if ctx.stockx_product else None,
            "stockx_confirmed": ctx.stockx_confirmed, "notes": resolution.notes,
        }
        if any(x in ctx.failure_codes for x in (scoring.CONTRADICTORY_EVIDENCE,)) and status == "high":
            status = "medium"
        review = status in ("low", "unresolved") or scoring.MULTIPLE_CANDIDATES in ctx.failure_codes \
            or scoring.STOCKX_MATCH_UNCERTAIN in ctx.failure_codes
        result = IdentifyResult(
            id=ctx.id, status=status, identified=status in ("high", "medium"), confidence=confidence,
            review_required=review, product=product, stockx=stockx, identification_method=ctx.methods,
            candidates=sorted(ctx.pool.all(), key=lambda c: (c.rejected, -c.score))[:12],
            evidence=ctx.evidence, failure_codes=ctx.failure_codes, stages=ctx.stages,
            input_images=ctx.image_paths, input_url=ctx.inp.product_url or None, input_title=ctx.inp.title or None,
            created_at=utcnow(),
        )
        ctx.result = result
        if self.db_enabled and winner is not None and status in ("high", "medium"):
            self._persist_product(winner, ctx)
        return result

    def _persist_product(self, winner: ScoredCandidate, ctx: RunContext) -> None:
        try:
            from app.db.session import db_session
            with db_session() as s:
                p = ctx.stockx_product
                repo.upsert_product(s, winner.style_code, name=winner.name, brand=winner.brand, model=winner.model,
                                    colorway=winner.colorway, gender=winner.gender, size_category=winner.size_category,
                                    release_date=winner.release_date, retail_price=winner.retail_price,
                                    stockx_product_id=p.product_id if p else winner.stockx_product_id,
                                    stockx_url_key=p.url_key if p else "", source="identification",
                                    display=winner.style_code_display)
        except Exception:
            logger.debug("persisting winner failed", exc_info=True)


class _Skip(Exception):
    pass


def _decodable(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def _dedupe_images(images: List[bytes]) -> List[bytes]:
    seen, out = set(), []
    for b in images:
        h = repo.sha256_bytes(b)
        if h in seen or not _decodable(b):
            continue
        seen.add(h)
        out.append(b)
        if len(out) >= settings.max_images_per_request:
            break
    return out


def store_images(ident_id: str, images: List[bytes]) -> List[str]:
    """Re-encodes uploads as JPEG under data/uploads/<id>/ for the review UI.
    Returns web paths."""
    folder = Path(settings.data_dir) / "uploads" / ident_id
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, data in enumerate(images):
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((1600, 1600))
            out = folder / f"{i}.jpg"
            img.save(out, format="JPEG", quality=88)
            paths.append(f"/uploads/{ident_id}/{i}.jpg")
        except Exception:
            continue
    return paths


def row_to_result(row) -> IdentifyResult:
    return IdentifyResult(
        id=row.id, status=row.status, identified=row.status in ("high", "medium"), confidence=row.confidence,
        review_required=bool(row.review_required),
        product=CanonicalProduct(**row.canonical_product) if row.canonical_product else None,
        stockx=StockXMatch(**row.stockx) if row.stockx else None,
        identification_method=row.identification_method or [],
        candidates=[ScoredCandidate(**c) for c in (row.candidates or [])],
        evidence=row.evidence or {}, failure_codes=row.failure_codes or [],
        stages=[Stage(**s) for s in (row.stages or [])], input_images=row.image_paths or [],
        input_url=row.input_url, input_title=row.input_title, created_at=row.created_at, error=row.error,
    )
