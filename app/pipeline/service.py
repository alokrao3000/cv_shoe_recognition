"""
Persistence layer around the pipeline: creates Identification rows, runs
the pipeline against them (updating after every stage so the UI can poll),
serves results, and records human review decisions as labeled data.
"""
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from sqlalchemy import select

from app.config import settings
from app.db import repo
from app.db.models import Identification, ReviewDecision, utcnow
from app.db.session import db_session
from app.extraction import sku_detector
from app.pipeline.identify import Pipeline, RunContext, RunInput, row_to_result, store_images
from app.pipeline import cache as rcache
from app.schemas import IdentifyResult, ReviewDecisionRequest, ReviewQueueItem

logger = logging.getLogger(__name__)


class IdentificationService:
    def __init__(self, pipeline: Optional[Pipeline] = None):
        self.pipeline = pipeline or Pipeline()

    # ── create / run ──

    def create(self, inp: RunInput) -> str:
        ident_id = str(uuid.uuid4())
        paths = store_images(ident_id, inp.images) if inp.images else []
        hashes = [repo.sha256_bytes(b) for b in inp.images]
        from urllib.parse import urlparse
        with db_session() as s:
            s.add(Identification(
                id=ident_id, status="queued",
                input_kind="mixed" if (inp.images or inp.image_urls) and inp.product_url else ("url" if inp.product_url else "image"),
                input_url=inp.product_url or None,
                input_domain=(urlparse(inp.product_url).hostname or "").lower().removeprefix("www.") if inp.product_url else None,
                input_title=inp.title or None, input_description=inp.description or None,
                image_hashes=hashes, image_paths=paths,
                cache_key=rcache.cache_key(hashes, inp.product_url, inp.title),
                evidence={"input": {"image_urls": inp.image_urls}},
                stages=[{"name": n, "label": l, "status": "pending"} for n, l in _stage_defs()],
            ))
        return ident_id

    def run(self, ident_id: str) -> None:
        with db_session() as s:
            row = s.get(Identification, ident_id)
            if row is None:
                return
            inp = RunInput(images=_load_images(row.image_paths), image_urls=(row.evidence or {}).get("input", {}).get("image_urls", []),
                           product_url=row.input_url or "", title=row.input_title or "", description=row.input_description or "")
            row.status = "running"

        def on_update(ctx: RunContext):
            with db_session() as s2:
                r = s2.get(Identification, ident_id)
                if r is not None:
                    r.stages = [st.model_dump() for st in ctx.stages]
                    r.failure_codes = list(ctx.failure_codes)

        try:
            result = self.pipeline.run_inline(inp, ident_id=ident_id, on_update=on_update)
        except Exception as exc:
            logger.exception(f"identification {ident_id} crashed")
            with db_session() as s:
                r = s.get(Identification, ident_id)
                r.status, r.error, r.finished_at = "error", f"{type(exc).__name__}: {exc}"[:500], utcnow()
            return
        self._store_result(ident_id, result)

    def _store_result(self, ident_id: str, result: IdentifyResult) -> None:
        with db_session() as s:
            r = s.get(Identification, ident_id)
            if r is None:
                return
            r.status = result.status
            r.confidence = result.confidence
            r.review_required = result.review_required
            r.selected_style_code = sku_detector.normalize_style_code(result.product.style_code) if result.product else None
            r.canonical_product = result.product.model_dump(mode="json") if result.product else None
            r.stockx = result.stockx.model_dump(mode="json") if result.stockx else None
            r.candidates = [c.model_dump(mode="json") for c in result.candidates]
            r.evidence = result.evidence
            r.failure_codes = result.failure_codes
            r.stages = [st.model_dump() for st in result.stages]
            r.identification_method = result.identification_method
            r.finished_at = utcnow()
            r.error = result.error
            if result.product and r.selected_style_code:
                product = repo.get_product_by_code(s, r.selected_style_code)
                r.selected_product_id = product.id if product else None

    def rerun(self, ident_id: str) -> Optional[str]:
        """New identification from the same inputs (bypasses the cache by id)."""
        with db_session() as s:
            row = s.get(Identification, ident_id)
            if row is None:
                return None
            inp = RunInput(images=_load_images(row.image_paths), product_url=row.input_url or "",
                           title=row.input_title or "", description=row.input_description or "")
        new_id = self.create(inp)
        with db_session() as s:
            s.get(Identification, new_id).cache_key = None
        return new_id

    # ── read ──

    def get(self, ident_id: str) -> Optional[IdentifyResult]:
        with db_session() as s:
            row = s.get(Identification, ident_id)
            return row_to_result(row) if row else None

    def recent(self, limit: int = 20) -> List[ReviewQueueItem]:
        with db_session() as s:
            rows = s.execute(select(Identification).order_by(Identification.created_at.desc()).limit(limit)).scalars().all()
            return [_queue_item(r) for r in rows]

    def queue(self, limit: int = 50) -> List[ReviewQueueItem]:
        with db_session() as s:
            rows = s.execute(select(Identification)
                             .where(Identification.review_required.is_(True), Identification.reviewed.is_(False))
                             .order_by(Identification.created_at.desc()).limit(limit)).scalars().all()
            return [_queue_item(r) for r in rows]

    # ── review ──

    def decide(self, ident_id: str, decision: ReviewDecisionRequest) -> Optional[IdentifyResult]:
        with db_session() as s:
            row = s.get(Identification, ident_id)
            if row is None:
                return None
            system_code = row.selected_style_code
            chosen: Optional[str] = None
            if decision.action == "confirm":
                chosen = system_code
            elif decision.action in ("select", "manual_sku"):
                hit = sku_detector.candidate_from_identifier(decision.style_code or "", "manual", "review")
                if hit is None:
                    raise ValueError("not a recognizable style code")
                chosen = hit.code
            s.add(ReviewDecision(identification_id=ident_id, action=decision.action, chosen_style_code=chosen,
                                 system_style_code=system_code, system_confidence=row.confidence, notes=decision.notes))
            row.reviewed = True
            row.review_required = False
            methods = list(row.identification_method or [])
            if "manual_review" not in methods:
                methods.append("manual_review")
            row.identification_method = methods
            if decision.action == "reject":
                row.status = "unresolved"
                row.selected_style_code = None
                row.selected_product_id = None
                row.stockx = None
                row.evidence = dict(row.evidence or {}, review={"action": "reject", "rejected_style_code": system_code})
            else:
                self._apply_human_match(s, row, chosen)
            s.flush()
            return row_to_result(row)

    def _apply_human_match(self, s, row: Identification, chosen: str) -> None:
        """A human verdict is definitive: set HIGH, attach StockX metadata,
        and label the query images as reference photos of the chosen product."""
        candidate = next((c for c in (row.candidates or []) if c.get("style_code") == chosen), None)
        product_meta = {}
        stockx_product = None
        if self.pipeline.catalog.available:
            try:
                hits = self.pipeline.catalog.by_style_code(chosen)
                if hits:
                    stockx_product = hits[0]
            except Exception:
                logger.debug("stockx lookup during review failed", exc_info=True)
        if stockx_product is not None:
            from app.extraction.normalize import parse_title
            parsed = parse_title(stockx_product.title, stockx_product.brand)
            product_meta = dict(name=stockx_product.title, brand=parsed.brand or stockx_product.brand, model=parsed.model,
                                colorway=stockx_product.colorway, gender=(stockx_product.gender or "").lower(),
                                release_date=stockx_product.release_date, retail_price=stockx_product.retail_price,
                                stockx_product_id=stockx_product.product_id, stockx_url_key=stockx_product.url_key)
        elif candidate:
            product_meta = {k: candidate.get(k) or "" for k in ("name", "brand", "model", "colorway", "gender", "size_category", "release_date")}
            product_meta["retail_price"] = candidate.get("retail_price")
            product_meta["stockx_product_id"] = candidate.get("stockx_product_id") or ""
        product = repo.upsert_product(s, chosen, source="user_confirmed", **product_meta)
        images = _load_images(row.image_paths)
        if images:
            try:
                vecs = self.pipeline.embed(images)
                for data, vec in zip(images, vecs):
                    repo.add_product_image(s, product, vec, image_url="", image_bytes=data, source="user_confirmed")
            except Exception:
                logger.exception("embedding confirmed images failed")
        row.status = "high"
        row.confidence = 1.0
        row.selected_style_code = chosen
        row.selected_product_id = product.id
        row.canonical_product = {
            "brand": product.brand or "", "model": product.model or "", "sub_model": product.sub_model or "",
            "colorway": product.colorway or "", "style_code": product.style_code_display or chosen,
            "gender": product.gender or "", "size_category": product.size_category or "",
            "release_date": product.release_date or "", "retailer_url": row.input_url or "",
            "product_name": product.name or "", "images": [],
        }
        if stockx_product is not None:
            market, err = self.pipeline.catalog.market(stockx_product)
            row.stockx = {
                "product_id": stockx_product.product_id, "url": stockx_product.url, "style_code": stockx_product.style_id,
                "product_name": stockx_product.title, "match_confidence": 1.0,
                "lowest_ask": market.lowest_ask if market else None, "highest_bid": market.highest_bid if market else None,
                "currency": "USD", "sizes": [sz.model_dump() for sz in market.sizes] if market else [], "market_error": err,
            }
        row.evidence = dict(row.evidence or {}, review={"action": "human_match", "chosen_style_code": chosen})


def _stage_defs():
    from app.pipeline.identify import STAGES
    return STAGES


def _load_images(paths: List[str]) -> List[bytes]:
    out = []
    for p in paths or []:
        fs = Path(settings.data_dir) / "uploads" / Path(p).parent.name / Path(p).name
        try:
            out.append(fs.read_bytes())
        except OSError:
            continue
    return out


def _queue_item(r: Identification) -> ReviewQueueItem:
    top = None
    if r.candidates:
        alive = [c for c in r.candidates if not c.get("rejected")]
        top = (alive or r.candidates)[0].get("style_code_display")
    return ReviewQueueItem(id=r.id, status=r.status, confidence=r.confidence, created_at=r.created_at,
                           input_title=r.input_title or (r.canonical_product or {}).get("product_name"),
                           input_url=r.input_url, input_images=r.image_paths or [],
                           top_candidate=(r.canonical_product or {}).get("style_code") or top,
                           failure_codes=r.failure_codes or [])
