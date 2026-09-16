"""Scoring / contradiction / tier logic on synthetic evidence — the spec's
required scenarios: exact SKU, missing SKU, similar sneakers, different
colorway, GS vs adult, ambiguous, incorrect-candidate rejection."""
import pytest

from app.config import settings
from app.pipeline import scoring as sc
from app.pipeline.candidates import CandidatePool
from app.schemas import (
    CandidateVerdict, EvidenceSource, ProductPageData, StockXProduct, StyleCodeCandidate, VerificationResult,
    VisionAnalysis,
)
from app.search.web_discovery import WebCandidate


def panda_stockx():
    return StockXProduct(product_id="p1", url_key="nike-dunk-low-retro-white-black-2021", style_id="DD1391-100",
                         style_codes=["DD1391100"], title="Nike Dunk Low Retro White Black Panda (2021)",
                         brand="Nike", colorway="White/Black", gender="men", release_date="2021-03-10",
                         retail_price=110.0, product_type="sneakers")


def reverse_panda_stockx():
    return StockXProduct(product_id="p2", url_key="nike-dunk-low-black-white", style_id="DJ6188-002",
                         style_codes=["DJ6188002"], title="Nike Dunk Low Reverse Panda", brand="Nike",
                         colorway="Black/White", gender="men", retail_price=110.0, product_type="sneakers")


def panda_gs_stockx():
    return StockXProduct(product_id="p3", url_key="nike-dunk-low-white-black-gs", style_id="CW1590-100",
                         style_codes=["CW1590100"], title="Nike Dunk Low Retro White Black (GS)", brand="Nike",
                         colorway="White/Black", gender="child", retail_price=90.0, product_type="sneakers")


def page_with_sku():
    return ProductPageData(title="Nike Dunk Low Retro White Black", brand="Nike", price=109.99,
                           sku_candidates=[StyleCodeCandidate(code="DD1391100", display="DD1391-100",
                                                              source="page:body", labeled=True, confidence=0.9)])


def vision_panda(**kw):
    base = dict(brand="Nike", model="Nike Dunk Low", colorway="White/Black", official_colorway_guess="Panda",
                gender="men", size_category="adult", brand_confidence=0.95, model_confidence=0.9,
                colorway_confidence=0.85)
    base.update(kw)
    return VisionAnalysis(**base)


def build(obs_page=None, vision=None, web=None, stockx_products=(), page_src=True, embeddings=None):
    pool = CandidatePool()
    page = obs_page
    if page and page_src:
        for c in page.sku_candidates:
            pool.add(c.code, EvidenceSource(kind="page_sku", detail=c.source, weight=c.confidence), display=c.display)
    if vision and vision.visible_style_code:
        pool.add(vision.visible_style_code, EvidenceSource(kind="vision_tag"))
    web = web or {}
    for wc in web.values():
        pool.add(wc.code, EvidenceSource(kind="web", detail=",".join(wc.domains)), display=wc.display)
    for p in stockx_products:
        pool.add_stockx_product(p, EvidenceSource(kind="stockx_search", detail="name"))
    for code, sim in (embeddings or {}).items():
        c = pool.add(code, EvidenceSource(kind="embedding", weight=sim))
        c.embedding_similarity = sim
    obs = sc.Observation(page=page, vision=vision, page_codes=page.sku_candidates if page else [], web_codes=web)
    cands = [sc.detect_contradictions(sc.score_candidate(c, obs), obs) for c in pool.all()]
    return cands, obs


def by_code(cands, code):
    return next(c for c in cands if c.style_code == code)


def test_exact_page_sku_confirmed_by_stockx_is_high():
    cands, obs = build(page_with_sku(), vision_panda(), stockx_products=[panda_stockx()],
                       embeddings={"DD1391100": 0.93})
    c = by_code(cands, "DD1391100")
    assert c.components["sku_match"] == 1.0
    assert c.components["model_match"] == 1.0
    assert c.components["colorway_match"] >= 0.9
    assert not c.rejected and c.contradictions == []
    res = sc.resolve(cands, obs, None, stockx_confirmed=True, verifier_available=False)
    assert res.status == "high" and res.winner.style_code == "DD1391100"


def test_missing_sku_web_agreement_caps_at_medium_without_verifier():
    page = ProductPageData(title="Nike Dunk Low Retro White / Black", brand="Nike", price=109.99)
    web = {"DD1391100": WebCandidate(code="DD1391100", display="DD1391-100", domains=["stockx.com", "goat.com", "nike.com"],
                                     trust=3.0)}
    cands, obs = build(page, vision_panda(), web=web, stockx_products=[panda_stockx()], embeddings={"DD1391100": 0.9})
    res = sc.resolve(cands, obs, None, stockx_confirmed=True, verifier_available=False)
    assert res.winner.style_code == "DD1391100"
    assert res.status == "medium"                       # HIGH needs the verifier when no page SKU
    assert any("verifier" in n for n in res.notes)


def test_verifier_same_lifts_to_high_and_different_rejects():
    page = ProductPageData(title="Nike Dunk Low Retro White / Black", brand="Nike", price=109.99)
    web = {"DD1391100": WebCandidate(code="DD1391100", display="DD1391-100", domains=["stockx.com", "goat.com", "nike.com"], trust=3.0)}
    cands, obs = build(page, vision_panda(), web=web, stockx_products=[panda_stockx(), reverse_panda_stockx()],
                       embeddings={"DD1391100": 0.9, "DJ6188002": 0.8})
    ver = VerificationResult(verdicts=[
        CandidateVerdict(style_code="DD1391100", verdict="same", confidence=0.9),
        CandidateVerdict(style_code="DJ6188002", verdict="different", confidence=0.9,
                         contradictions=["query has white base, candidate is black base"]),
    ])
    sc.apply_verification(cands, ver)
    res = sc.resolve(cands, obs, ver, stockx_confirmed=True, verifier_available=True)
    assert res.status == "high" and res.winner.style_code == "DD1391100"
    assert by_code(cands, "DJ6188002").rejected


def test_reverse_colorway_scores_lower_and_is_flagged():
    cands, obs = build(page_with_sku(), vision_panda(), stockx_products=[panda_stockx(), reverse_panda_stockx()],
                       page_src=False, embeddings={"DD1391100": 0.9, "DJ6188002": 0.86})
    panda, reverse = by_code(cands, "DD1391100"), by_code(cands, "DJ6188002")
    assert panda.components["colorway_match"] > reverse.components["colorway_match"]
    assert panda.score > reverse.score


def test_gs_candidate_rejected_for_adult_query():
    page = ProductPageData(title="Nike Dunk Low Retro White Black Men's", brand="Nike", price=109.99)
    cands, obs = build(page, vision_panda(), stockx_products=[panda_stockx(), panda_gs_stockx()],
                       embeddings={"DD1391100": 0.9, "CW1590100": 0.91})
    gs = by_code(cands, "CW1590100")
    assert gs.rejected and gs.rejection_reason.startswith("size_category_mismatch")
    res = sc.resolve(cands, obs, None, stockx_confirmed=True, verifier_available=False)
    assert res.winner.style_code == "DD1391100"


def test_gs_query_prefers_gs_candidate():
    page = ProductPageData(title="Nike Dunk Low (GS) White Black", brand="Nike", price=89.99)
    cands, obs = build(page, vision_panda(size_category="gs"), stockx_products=[panda_stockx(), panda_gs_stockx()],
                       embeddings={"DD1391100": 0.9, "CW1590100": 0.9})
    assert by_code(cands, "DD1391100").rejected
    assert not by_code(cands, "CW1590100").rejected


def test_model_mismatch_rejects():
    high = StockXProduct(product_id="p4", url_key="nike-dunk-high-panda", style_id="DD1399-105",
                         style_codes=["DD1399105"], title="Nike Dunk High Retro White Black Panda", brand="Nike",
                         colorway="White/Black", gender="men", product_type="sneakers")
    cands, obs = build(ProductPageData(title="Nike Dunk Low Retro White Black"), vision_panda(),
                       stockx_products=[panda_stockx(), high], embeddings={"DD1391100": 0.9, "DD1399105": 0.92})
    assert by_code(cands, "DD1399105").rejected
    assert by_code(cands, "DD1399105").rejection_reason.startswith("model_mismatch")


def test_ambiguous_top_two_flags_multiple_candidates():
    a = panda_stockx()
    b = StockXProduct(product_id="p5", url_key="nike-dunk-low-retro-white-black-2024", style_id="FD9064-011",
                      style_codes=["FD9064011"], title="Nike Dunk Low Retro White Black Panda (2024)", brand="Nike",
                      colorway="White/Black", gender="men", retail_price=115.0, product_type="sneakers")
    cands, obs = build(ProductPageData(title="Nike Dunk Low Retro White Black", price=110), vision_panda(),
                       stockx_products=[a, b], embeddings={"DD1391100": 0.9, "FD9064011": 0.9})
    res = sc.resolve(cands, obs, None, stockx_confirmed=True, verifier_available=False)
    assert sc.MULTIPLE_CANDIDATES in res.failure_codes
    # Name + visual agreement with no style-code evidence can never be confident.
    assert res.status in ("low", "unresolved") and res.confidence < settings.confidence_medium


def test_no_candidates_is_unresolved():
    res = sc.resolve([], sc.Observation(), None, stockx_confirmed=False, verifier_available=False)
    assert res.status == "unresolved" and res.failure_codes == [sc.NO_CANDIDATES]


def test_tag_code_mismatch_penalizes_and_page_code_bonus():
    vis = vision_panda(visible_style_code="DD1391-100")
    cands, obs = build(ProductPageData(title="Nike Dunk Low Retro"), vis,
                       stockx_products=[panda_stockx(), reverse_panda_stockx()])
    rev = by_code(cands, "DJ6188002")
    assert any(x.startswith("tag_style_code_mismatch") for x in rev.contradictions)
    assert by_code(cands, "DD1391100").components["sku_match"] == 0.9


def test_verifier_uncertain_caps_below_high():
    cands, obs = build(page_with_sku(), vision_panda(), stockx_products=[panda_stockx()], embeddings={"DD1391100": 0.95})
    sc.apply_verification(cands, VerificationResult(verdicts=[CandidateVerdict(style_code="DD1391100", verdict="uncertain")]))
    res = sc.resolve(cands, obs, None, stockx_confirmed=True, verifier_available=True)
    assert res.status != "high"


def test_apparel_product_type_is_rejected():
    tee = StockXProduct(product_id="p9", url_key="nike-tee", style_id="DD1391-100", style_codes=["DD1391100"],
                        title="Nike Panda Tee", brand="Nike", product_type="apparel")
    cands, obs = build(ProductPageData(title="Nike Dunk Low Retro"), vision_panda(), stockx_products=[tee])
    assert by_code(cands, "DD1391100").rejected


def test_score_is_within_unit_interval_and_weights_sum_to_one():
    w = sc.weights()
    assert pytest.approx(sum(w.values()), abs=1e-6) == 1.0
    cands, _ = build(page_with_sku(), vision_panda(), stockx_products=[panda_stockx()], embeddings={"DD1391100": 0.99})
    assert 0.0 <= cands[0].score <= 1.0
