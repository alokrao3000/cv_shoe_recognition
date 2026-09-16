"""End-to-end pipeline on fakes: no network, no DB, no torch."""
import io

import numpy as np
from PIL import Image

from app.pipeline import scoring
from app.pipeline.identify import Pipeline, RunInput
from app.schemas import CandidateVerdict, StockXProduct, VerificationResult, VisionAnalysis
from app.search.provider import SearchError, WebResult
from app.stockx.client import StockXRequestFailed
from app.vision.claude_vision import VisionError


def jpeg(color=(240, 240, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="JPEG")
    return buf.getvalue()


PANDA = StockXProduct(product_id="p1", url_key="nike-dunk-low-retro-white-black-2021", style_id="DD1391-100",
                      style_codes=["DD1391100"], title="Nike Dunk Low Retro White Black Panda (2021)", brand="Nike",
                      colorway="White/Black", gender="men", release_date="2021-03-10", retail_price=110.0,
                      product_type="sneakers")
REVERSE = StockXProduct(product_id="p2", url_key="nike-dunk-low-reverse-panda", style_id="DJ6188-002",
                        style_codes=["DJ6188002"], title="Nike Dunk Low Reverse Panda", brand="Nike",
                        colorway="Black/White", gender="men", retail_price=110.0, product_type="sneakers")


class FakeCatalog:
    available = True

    def __init__(self, products=(PANDA, REVERSE), fail=False):
        self.products = list(products)
        self.fail = fail
        self.calls = []

    def search(self, query, page_size=20):
        self.calls.append(("search", query))
        if self.fail:
            raise StockXRequestFailed("/catalog/search", 503, "down", 1)
        return list(self.products)

    def by_style_code(self, code):
        self.calls.append(("style", code))
        if self.fail:
            raise StockXRequestFailed("/catalog/search", 503, "down", 1)
        norm = code.replace("-", "").upper()
        return [p for p in self.products if norm in p.style_codes]

    def market(self, product):
        from app.schemas import MarketData, SizeMarket
        return MarketData(product_id=product.product_id, url=product.url, lowest_ask=125.0, highest_bid=110.0,
                          sizes=[SizeMarket(size="10", lowest_ask=125.0, highest_bid=110.0)]), None


class FakeProvider:
    name = "fake"
    available = True

    def __init__(self, fail=False):
        self.fail = fail

    def web_search(self, query, num=10):
        if self.fail:
            raise SearchError("http_500")
        return [WebResult("Nike Dunk Low Retro White Black DD1391-100 | StockX", "https://stockx.com/x", ""),
                WebResult("Nike Dunk Low 'Panda' (DD1391-100) Release", "https://sneakernews.com/x", "White/Black"),
                WebResult("Buy Nike Dunk Low Panda DD1391-100", "https://www.goat.com/x", "")]

    def image_search(self, query, num=10):
        return []


class FakeVision:
    available = True

    def __init__(self, analysis=None, verification=None, fail=False):
        self.analysis = analysis or VisionAnalysis(
            brand="Nike", model="Nike Dunk Low", colorway="White/Black", official_colorway_guess="Panda",
            gender="men", size_category="adult", brand_confidence=0.95, model_confidence=0.9, colorway_confidence=0.85,
            likely_release_names=["Nike Dunk Low Retro White Black Panda"], likely_style_codes=["DD1391-100"])
        self.verification = verification
        self.fail = fail

    def analyze(self, images, context=None):
        if self.fail:
            raise VisionError("rate_limited")
        return self.analysis

    def verify(self, images, candidates, reference_images=None):
        if self.fail:
            raise VisionError("rate_limited")
        return self.verification or VerificationResult(verdicts=[
            CandidateVerdict(style_code=c.style_code, verdict="same" if c.style_code == "DD1391100" else "different",
                             confidence=0.9) for c in candidates])


def fake_embed(images):
    return [np.ones(768, dtype="float32") / np.sqrt(768) for _ in images]


def make_pipeline(**kw):
    return Pipeline(catalog=kw.get("catalog", FakeCatalog()), provider=kw.get("provider", FakeProvider()),
                    vision=kw.get("vision", FakeVision()), embed_fn=fake_embed, db_enabled=False)


def test_full_run_image_plus_title_is_high_and_mapped():
    res = make_pipeline().run_inline(RunInput(images=[jpeg()], title="Nike Dunk Low Retro White / Black"))
    assert res.status == "high" and res.identified
    assert res.product.style_code == "DD1391-100"
    assert res.stockx and res.stockx.product_id == "p1" and res.stockx.lowest_ask == 125.0
    assert res.stockx.url == "https://stockx.com/nike-dunk-low-retro-white-black-2021"
    assert {"computer_vision", "web_search", "stockx_match", "visual_verification"} <= set(res.identification_method)
    assert all(s.status in ("done", "skipped") for s in res.stages)
    assert "resolution" in res.evidence and res.evidence["resolution"]["stockx_confirmed"]
    rejected = [c for c in res.candidates if c.style_code == "DJ6188002"]
    assert rejected and rejected[0].rejected


def test_failed_web_search_and_stockx_are_recorded_not_fatal():
    res = make_pipeline(provider=FakeProvider(fail=True), catalog=FakeCatalog(fail=True)).run_inline(
        RunInput(images=[jpeg()], title="Nike Dunk Low Retro White / Black"))
    assert scoring.WEB_SEARCH_FAILED in res.failure_codes
    assert scoring.STOCKX_API_FAILED in res.failure_codes
    assert res.status != "high"
    assert res.stockx is None
    assert res.stages[-1].status == "skipped"


def test_no_vision_key_degrades_gracefully():
    v = FakeVision()
    v.available = False
    res = make_pipeline(vision=v).run_inline(RunInput(images=[jpeg()], title="Nike Dunk Low Retro White / Black"))
    assert scoring.VISION_UNAVAILABLE in res.failure_codes
    assert res.status in ("medium", "low", "unresolved")     # never HIGH without verification or page SKU
    assert "computer_vision" not in res.identification_method


def test_vision_error_is_uncertain_not_crash():
    res = make_pipeline(vision=FakeVision(fail=True)).run_inline(RunInput(images=[jpeg()], title="Nike Dunk Low Retro"))
    assert scoring.VISION_UNCERTAIN in res.failure_codes
    assert res.stages[3].status == "done" and "failed" in res.stages[3].detail


def test_not_a_sneaker_short_circuits():
    v = FakeVision(analysis=VisionAnalysis(is_sneaker=False))
    res = make_pipeline(vision=v).run_inline(RunInput(images=[jpeg()]))
    assert res.status == "unresolved" and scoring.NOT_A_SNEAKER in res.failure_codes
    assert res.review_required


def test_verifier_uncertain_everything_sends_to_review():
    ver = VerificationResult(verdicts=[CandidateVerdict(style_code="DD1391100", verdict="uncertain", confidence=0.5),
                                       CandidateVerdict(style_code="DJ6188002", verdict="uncertain", confidence=0.5)])
    res = make_pipeline(vision=FakeVision(verification=ver)).run_inline(
        RunInput(images=[jpeg()], title="Nike Dunk Low Retro White / Black"))
    assert res.status != "high"


def test_title_only_input_without_images():
    res = make_pipeline().run_inline(RunInput(title="Nike Dunk Low Retro White Black Style: DD1391-100"))
    assert res.product and res.product.style_code == "DD1391-100"
    assert res.stages[3].status == "skipped"       # vision skipped: no images
    assert res.status in ("high", "medium")


def test_empty_input_fails_ingest():
    res = make_pipeline().run_inline(RunInput())
    assert res.stages[0].status == "failed"
    assert res.status == "unresolved"
