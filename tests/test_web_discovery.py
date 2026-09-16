from app.schemas import ProductPageData, VisionAnalysis
from app.search import web_discovery as wd
from app.search.provider import SearchError, WebResult


class FakeProvider:
    name = "fake"
    available = True

    def __init__(self, results=None, fail=False):
        self.results = results or {}
        self.fail = fail
        self.queries = []

    def web_search(self, query, num=10):
        self.queries.append(query)
        if self.fail:
            raise SearchError("http_500")
        return self.results.get(query, [])

    def image_search(self, query, num=10):
        return []


def test_build_queries_uses_page_and_vision():
    page = ProductPageData(title="Nike Dunk Low Retro White Black", brand="Nike")
    vision = VisionAnalysis(brand="Nike", model="Nike Dunk Low", colorway="White/Black",
                            official_colorway_guess="Panda", likely_release_names=["Nike Dunk Low Retro Panda"])
    qs = wd.build_queries(page, vision, partial_codes=["DD1391-100"])
    assert qs[0] == "Nike Dunk Low Retro White Black style code"
    assert any("Panda" in q for q in qs)
    assert "DD1391-100" in qs
    assert len(qs) <= 6


def test_discover_aggregates_agreement_across_domains():
    q = "Nike Dunk Low Retro White Black style code"
    provider = FakeProvider({q: [
        WebResult("Nike Dunk Low Retro White Black DD1391-100 - StockX", "https://stockx.com/nike-dunk-low-retro-white-black", "Style DD1391-100"),
        WebResult("Nike Dunk Low 'Panda' DD1391-100", "https://www.sneakernews.com/panda", "colorway White/Black"),
        WebResult("Dunk Low Panda DD1391-100 | GOAT", "https://www.goat.com/sneakers/dunk-low", ""),
        WebResult("Some blog: my favourite Dunks", "https://blog.example.com/x", "I like DD1503-101 and DD1391-100"),
    ]})
    cands, errors = wd.discover([q], provider, brand_hint="nike")
    assert errors == []
    assert cands[0].code == "DD1391100"
    assert cands[0].agreement == 4
    assert "stockx.com" in cands[0].domains
    assert cands[0].trust > 2.5
    other = [c for c in cands if c.code == "DD1503101"]
    assert other and other[0].agreement == 1 and other[0].trust < cands[0].trust


def test_discover_reports_errors_and_unavailable_provider():
    cands, errors = wd.discover(["x"], FakeProvider(fail=True))
    assert cands == [] and errors and "http_500" in errors[0]
    p = FakeProvider()
    p.available = False
    assert wd.discover(["x"], p) == ([], ["no_provider"])
