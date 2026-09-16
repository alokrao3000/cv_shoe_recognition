from app.extraction import sku_detector as sd


def codes(cands):
    return [c.code for c in cands]


def test_normalize_equivalents():
    assert sd.normalize_style_code("DD1391-100") == "DD1391100"
    assert sd.normalize_style_code("dd1391 100") == "DD1391100"
    assert sd.codes_equivalent("DD1391-100", "DD1391100")


def test_labeled_nike_code_is_high_confidence():
    cands = sd.detect_style_codes("Nike Dunk Low Retro. Style: DD1391-100. Color: White/Black", "page:body")
    assert codes(cands)[0] == "DD1391100"
    assert cands[0].labeled and cands[0].confidence >= 0.85
    assert cands[0].display == "DD1391-100"


def test_bare_nike_code_with_dash_is_accepted():
    cands = sd.detect_style_codes("Air Jordan 4 Retro Bred Reimagined FV5029-006 men's", "page:title")
    assert "FV5029006" in codes(cands)


def test_bare_nike_code_without_dash_needs_brand_context():
    assert sd.detect_style_codes("Promo DD1391100 in stock", "page:body") == []
    cands = sd.detect_style_codes("Nike Dunk Low DD1391100 in stock", "page:body")
    assert "DD1391100" in codes(cands)


def test_dates_prices_upcs_are_rejected():
    text = "Released 2024-03-15, was $120.00, UPC 196155451231, order 20240315"
    assert sd.detect_style_codes(text, "page:body") == []


def test_adidas_code_requires_hint_or_label():
    assert sd.detect_style_codes("Some product GX8862 here", "page:body") == []
    assert "GX8862" in codes(sd.detect_style_codes("adidas Samba OG GX8862", "page:title"))
    assert "IE1234" in codes(sd.detect_style_codes("Article number: IE1234", "page:body"))


def test_new_balance_and_asics_formats():
    assert "M990GL6" in codes(sd.detect_style_codes("New Balance 990v6 M990GL6", "page:title"))
    assert "1201A019020" in codes(sd.detect_style_codes("ASICS Gel-Kayano 14 1201A019-020", "page:title"))
    assert "U9060ECC" in codes(sd.detect_style_codes("New Balance 9060 Sea Salt U9060ECC", "page:title"))


def test_other_brand_formats_with_label():
    assert "VN0A38G1EO2" in codes(sd.detect_style_codes("Vans Old Skool SKU: VN0A38G1EO2", "page:body"))
    assert "162050C" in codes(sd.detect_style_codes("Converse Chuck 70 Style # 162050C", "page:body"))
    assert "L47293900" in codes(sd.detect_style_codes("Salomon XT-6 style code L47293900", "page:body"))


def test_wrong_brand_shape_on_branded_page_is_dropped():
    # An adidas-shaped token on a Nike page with no label is not a Nike SKU.
    cands = sd.detect_style_codes("Nike Air Force 1 '07 white promo GX8862 shipping", "page:body")
    assert "GX8862" not in codes(cands)


def test_identifier_strips_size_suffix():
    c = sd.candidate_from_identifier("DV0831-101-10.5", "variant_sku", "page", "nike")
    assert c is not None and c.code == "DV0831101" and c.display == "DV0831-101"
    assert sd.candidate_from_identifier("12345", "sku", "page") is None


def test_merge_keeps_best_confidence():
    a = sd.detect_style_codes("Dunk Low DD1391-100", "page:title", base_confidence=0.6)
    b = sd.detect_style_codes("Style: DD1391-100", "page:body", base_confidence=0.45)
    merged = sd.merge_candidates([a, b])
    assert len(merged) == 1 and merged[0].labeled
