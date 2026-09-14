import numpy as np
import pytest

from app.index import ReferenceIndex, ReferenceItem, classify_match
from app.models import Candidate


def _unit(vec):
    return vec / np.linalg.norm(vec)


@pytest.fixture
def small_index():
    items = [
        ReferenceItem(sku="AAA-111", name="Shoe A", image_url="http://x/a.jpg"),
        ReferenceItem(sku="BBB-222", name="Shoe B", image_url="http://x/b.jpg"),
        ReferenceItem(sku="CCC-333", name="Shoe C", image_url="http://x/c.jpg"),
    ]
    vectors = np.stack([
        _unit(np.array([1.0, 0.0, 0.0], dtype="float32")),
        _unit(np.array([0.0, 1.0, 0.0], dtype="float32")),
        _unit(np.array([0.0, 0.0, 1.0], dtype="float32")),
    ])
    return ReferenceIndex.build(items, vectors)


def test_search_returns_exact_match_first(small_index):
    query = _unit(np.array([1.0, 0.01, 0.0], dtype="float32"))
    results = small_index.search(query, k=3)
    assert results[0].sku == "AAA-111"
    assert results[0].similarity == pytest.approx(1.0, abs=0.05)


def test_search_respects_k(small_index):
    query = _unit(np.array([1.0, 0.0, 0.0], dtype="float32"))
    assert len(small_index.search(query, k=2)) == 2


def test_save_and_load_roundtrip(small_index, tmp_path):
    small_index.save(str(tmp_path))
    loaded = ReferenceIndex.load(str(tmp_path))
    assert len(loaded) == len(small_index)

    query = _unit(np.array([0.0, 1.0, 0.0], dtype="float32"))
    assert loaded.search(query, k=1)[0].sku == "BBB-222"


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReferenceIndex.load(str(tmp_path / "does_not_exist"))


def test_merged_with_adds_new_items_without_touching_existing(small_index):
    new_items = [ReferenceItem(sku="DDD-444", name="Shoe D", image_url="http://x/d.jpg")]
    new_vec = _unit(np.array([0.0, 0.0, -1.0], dtype="float32")).reshape(1, -1)

    merged = small_index.merged_with(new_items, new_vec)

    assert len(merged) == 4
    assert merged.skus == {"AAA-111", "BBB-222", "CCC-333", "DDD-444"}
    assert len(small_index) == 3   # original untouched

    query = _unit(np.array([0.0, 0.0, -1.0], dtype="float32"))
    assert merged.search(query, k=1)[0].sku == "DDD-444"


def test_merged_with_empty_returns_self(small_index):
    merged = small_index.merged_with([], np.empty((0, 3), dtype="float32"))
    assert merged is small_index


def test_merged_with_mismatched_lengths_raises(small_index):
    with pytest.raises(ValueError):
        small_index.merged_with(
            [ReferenceItem(sku="X", name=None, image_url=None)],
            np.empty((0, 3), dtype="float32"),
        )


def test_classify_match_confident():
    candidates = [Candidate(sku="A", similarity=0.95), Candidate(sku="B", similarity=0.60)]
    identified, reason = classify_match(candidates)
    assert identified
    assert reason is None


def test_classify_match_low_similarity():
    candidates = [Candidate(sku="A", similarity=0.40)]
    identified, reason = classify_match(candidates)
    assert not identified
    assert reason == "no_confident_match"


def test_classify_match_ambiguous():
    candidates = [Candidate(sku="A", similarity=0.90), Candidate(sku="B", similarity=0.895)]
    identified, reason = classify_match(candidates)
    assert not identified
    assert reason == "ambiguous_top_match"


def test_classify_match_empty():
    identified, reason = classify_match([])
    assert not identified
    assert reason == "no_candidates"
