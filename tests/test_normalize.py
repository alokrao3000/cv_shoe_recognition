import pytest

from app.extraction import normalize as n


def test_canonical_brand_aliases():
    assert n.canonical_brand("Nike SB") == "Nike"
    assert n.canonical_brand("Air Jordan") == "Jordan"
    assert n.canonical_brand("adidas Originals") == "adidas"
    assert n.canonical_brand("NEW BALANCE") == "New Balance"
    assert n.brand_family("Jordan") == "Nike"


@pytest.mark.parametrize("title,expected", [
    ("Nike Air Jordan 1 Retro High OG 'Chicago'", "Air Jordan 1 High"),
    ("Jordan 1 Retro High OG", "Air Jordan 1 High"),
    ("Air Jordan 1 Low SE", "Air Jordan 1 Low"),
    ("AJ1 High Bred Toe", "Air Jordan 1 High"),
    ("Air Jordan 1 Little Kids Retro Low \"Chicago\" Shoes", "Air Jordan 1 Low"),
    ("Jordan 1 Retro Low OG Chicago (2025) (PS)", "Air Jordan 1 Low"),
    ("Air Jordan 1 Retro (GS) Black Toe", "Air Jordan 1 High"),
    ("Jordan 1 Mid SE (PS)", "Air Jordan 1 Mid"),
    ("Nike Dunk Little Kids Low Panda", "Nike Dunk Low"),
    ("Nike Air Force 1 Mid '07", "Nike Air Force 1 Mid"),
    ("Nike Air Force 1 '07 LV8", "Nike Air Force 1 Low"),
    ("Air Jordan 11 Retro Low (GS)", "Air Jordan 11 Low"),
    ("Nike Dunk Low Retro White Black Panda", "Nike Dunk Low"),
    ("Nike SB Dunk Low Pro Strangelove", "Nike SB Dunk Low"),
    ("Nike Dunk High Retro", "Nike Dunk High"),
    ("Nike Air Force 1 '07", "Nike Air Force 1 Low"),
    ("adidas Samba OG Cloud White", "adidas Samba OG"),
    ("New Balance 990v6 Grey", "New Balance 990v6"),
    ("New Balance 9060 Sea Salt", "New Balance 9060"),
    ("ASICS Gel-Kayano 14 Cream", "ASICS Gel-Kayano 14"),
    ("Yeezy Boost 350 V2 Bone", "adidas Yeezy Boost 350 V2"),
])
def test_canonical_model(title, expected):
    assert n.canonical_model(title)[0] == expected


def test_sub_model_tokens():
    model, sub = n.canonical_model("Air Jordan 1 Retro High OG")
    assert model == "Air Jordan 1 High" and "retro" in sub and "og" in sub


@pytest.mark.parametrize("text,gender,cat", [
    ("Nike Dunk Low (GS)", "kids", "gs"),
    ("Nike Dunk Low Grade School", "kids", "gs"),
    ("Jordan 4 Retro (PS)", "kids", "ps"),
    ("Jordan 4 Retro Toddler", "kids", "td"),
    ("Nike Dunk Low Women's", "women", "adult"),
    ("Nike Dunk Low WMNS", "women", "adult"),
    ("Nike Dunk Low Men's", "men", "adult"),
    ("Nike Dunk Low Retro", "unknown", "unknown"),
])
def test_gender_and_category(text, gender, cat):
    assert n.parse_gender_and_category(text) == (gender, cat)


def test_compatibility_helpers():
    assert n.category_compatible("gs", "adult") is False
    assert n.category_compatible("gs", "unknown") is None
    assert n.gender_compatible("women", "men") is False
    assert n.gender_compatible("women", "unisex") is None


def test_colorway_tokens():
    assert n.colorway_tokens("White/Black-Gum") == ["white", "black", "gum"]
    assert n.colorway_tokens("Sail/University Red") == ["sail", "university red"]
    assert n.colorway_tokens("WHT/BLK") == ["white", "black"]


def test_colorway_similarity_is_order_sensitive_on_lead_color():
    same = n.colorway_similarity("White/Black", "White/Black")
    reversed_ = n.colorway_similarity("White/Black", "Black/White")
    different = n.colorway_similarity("White/Black", "Sail/University Red")
    assert same == pytest.approx(1.0)
    assert reversed_ < same
    assert different < reversed_
    assert n.colorway_similarity("", "White") is None


def test_model_similarity():
    assert n.model_similarity("Nike Dunk Low Panda", "Nike Dunk Low Retro") == 1.0
    assert n.model_similarity("Nike Dunk Low", "Nike Dunk High") < 0.7
    assert n.model_similarity("Nike Dunk", "Nike Dunk Low") == 0.6
    assert n.model_similarity("", "x") is None


def test_parse_title():
    p = n.parse_title("Nike Dunk Low Retro White/Black (GS)")
    assert p.brand == "Nike" and p.model == "Nike Dunk Low"
    assert p.size_category == "gs"
    assert "white" in p.colorway_text.lower() and "black" in p.colorway_text.lower()

    p2 = n.parse_title("Travis Scott x Air Jordan 1 Low OG 'Mocha'")
    assert p2.brand == "Jordan" and p2.model == "Air Jordan 1 Low"
    assert p2.collab.lower() == "travis scott"
    assert p2.colorway_text == "Mocha"
