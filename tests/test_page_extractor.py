import json
from pathlib import Path

import pytest

from app.extraction import page_extractor as pe

FIX = Path(__file__).parent / "fixtures"


def test_jsonld_page_extracts_everything():
    html = (FIX / "jsonld_page.html").read_text(encoding="utf-8")
    page = pe.extract_from_html(html, "https://www.somestore.com/p/aj4-bred")
    assert page.title.startswith("Air Jordan 4 Retro")
    assert page.brand == "Jordan"
    assert page.price == 215.0 and page.currency == "USD"
    assert page.availability == "InStock"
    assert page.breadcrumbs == ["Home", "Men", "Jordan"]
    assert "jsonld" in page.extraction_sources
    assert page.sku_candidates and page.sku_candidates[0].code == "FV5029006"
    assert page.sku_candidates[0].labeled
    assert page.raw_identifiers["color"].startswith("Black/Fire Red")
    imgs = page.images
    assert "https://images.somestore.com/aj4-bred-1.jpg" in imgs
    assert "https://images.somestore.com/aj4-bred-3.jpg" in imgs
    assert not any("logo" in u for u in imgs)


def test_shopify_json_is_merged():
    data = json.loads((FIX / "shopify_product.json").read_text(encoding="utf-8"))["product"]
    html = "<html><head><title>x</title></head><body><h1>Nike Dunk Low Retro</h1></body></html>"
    page = pe.extract_from_html(html, "https://shop.example.com/products/nike-dunk-low-retro-white-black",
                                shopify_product=data)
    assert "shopify_json" in page.extraction_sources
    assert page.brand == "Nike"
    assert page.price == 109.99
    assert page.sizes == ["9", "10"]
    assert len(page.images) == 2
    top = page.sku_candidates[0]
    assert top.code == "DD1391100" and top.display == "DD1391-100"
    assert "barcode" in page.raw_identifiers


def test_nextjs_embedded_state():
    html = (FIX / "nextjs_page.html").read_text(encoding="utf-8")
    page = pe.extract_from_html(html, "https://www.retailer.com/product/nb-9060")
    assert "embedded_json" in page.extraction_sources
    assert page.sku_candidates[0].code == "U9060ECC"
    assert "https://cdn.retailer.com/nb9060-1.jpg" in page.images
    assert "https://cdn.retailer.com/nb9060-2.webp" in page.images


def test_no_sku_page_yields_metadata_but_no_candidates():
    html = (FIX / "no_sku_page.html").read_text(encoding="utf-8")
    page = pe.extract_from_html(html, "https://boutique.example.com/nike-dunk-low")
    assert page.title == "Nike Dunk Low Retro"
    assert page.sku_candidates == []          # "20240915" must not become a SKU
    assert len(page.images) == 3
    assert page.error is None


def test_private_urls_are_blocked():
    page = pe.extract_product_page("http://127.0.0.1:8100/admin")
    assert page.error and page.error.startswith("blocked")
    page = pe.extract_product_page("ftp://example.com/x")
    assert page.error and page.error.startswith("blocked")
