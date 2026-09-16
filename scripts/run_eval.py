"""
Runs the identification pipeline over data/eval/cases.jsonl and reports:
top-1 / top-3 candidate accuracy, exact style-code accuracy, StockX-mapping
accuracy, the false-positive rate (confident but wrong — the headline
metric), unresolved rate, and latency. Ablations: --no-llm, --no-web,
--no-db turn signals off so each one's contribution is measurable.

Usage:
    python scripts/run_eval.py [--limit N] [--tag gs] [--no-llm] [--no-web] [--out data/eval/results.jsonl]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                                  # noqa: E402
from app.extraction.sku_detector import normalize_style_code     # noqa: E402
from app.pipeline.identify import Pipeline, RunInput             # noqa: E402
from app.search.provider import NullProvider                     # noqa: E402
from app.vision.claude_vision import VisionClient                # noqa: E402


class _NoVision(VisionClient):
    @property
    def available(self):
        return False


def _cand_summary(c):
    if c is None:
        return None
    return {"style_code": c.style_code_display, "name": c.name, "score": c.score, "rejected": c.rejected,
            "rejection_reason": c.rejection_reason, "components": c.components, "component_notes": c.component_notes,
            "contradictions": c.contradictions, "sources": sorted({s.kind for s in c.sources}),
            "embedding_similarity": c.embedding_similarity,
            "verification": c.verification.model_dump() if c.verification else None}


def load_cases(path: Path, limit, tag):
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        c = json.loads(line)
        if tag and tag not in (c.get("tags") or []):
            continue
        cases.append(c)
    return cases[:limit] if limit else cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(Path(settings.data_dir) / "eval" / "cases.jsonl"))
    ap.add_argument("--out", default=str(Path(settings.data_dir) / "eval" / "results.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--no-url", action="store_true", help="Ignore product_url (image + title only)")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases), args.limit, args.tag)
    if not cases:
        sys.exit("no cases — run scripts/build_eval_set.py or add data/eval/curated.jsonl entries")
    pipeline = Pipeline(provider=NullProvider() if args.no_web else None,
                        vision=_NoVision() if args.no_llm else None, db_enabled=not args.no_db)

    results, latencies = [], []
    confident_wrong = confident = 0
    top1 = top3 = exact = mapped = unresolved = 0
    for i, c in enumerate(cases, 1):
        images = []
        if c.get("image_path"):
            p = Path(settings.data_dir) / "eval" / "images" / c["image_path"]
            if p.exists():
                images.append(p.read_bytes())
        inp = RunInput(images=images, image_urls=[c["image_url"]] if c.get("image_url") else [],
                       product_url="" if args.no_url else c.get("product_url", ""), title=c.get("title", ""),
                       description=c.get("description", ""))
        t0 = time.perf_counter()
        r = pipeline.run_inline(inp)
        ms = int((time.perf_counter() - t0) * 1000)
        latencies.append(ms)
        expected = normalize_style_code(c["expected_style_code"])
        got = normalize_style_code(r.product.style_code) if r.product else ""
        ranked = [x.style_code for x in r.candidates if not x.rejected]
        hit1 = got == expected
        hit3 = expected in ranked[:3]
        top1 += hit1
        top3 += hit3
        if r.status in ("high", "medium"):
            confident += 1
            exact += hit1
            if not hit1:
                confident_wrong += 1
            if hit1 and r.stockx and (not c.get("expected_stockx_product_id") or r.stockx.product_id == c["expected_stockx_product_id"]):
                mapped += 1
        if r.status == "unresolved":
            unresolved += 1
        winner = next((x for x in r.candidates if not x.rejected), None)
        expected_cand = next((x for x in r.candidates if x.style_code == expected), None)
        vis = r.evidence.get("vision") or {}
        row = {"expected": c["expected_style_code"], "got": r.product.style_code if r.product else None, "status": r.status,
               "confidence": r.confidence, "top1": hit1, "top3": hit3, "ms": ms, "failure_codes": r.failure_codes,
               "tags": c.get("tags", []), "title": c.get("title"), "stockx": r.stockx.product_id if r.stockx else None,
               "ranking": ranked[:5], "notes": (r.evidence.get("resolution") or {}).get("notes", []),
               "errors": r.evidence.get("errors", {}),
               "vision": {k: vis.get(k) for k in ("brand", "model", "colorway", "official_colorway_guess", "gender",
                                                  "size_category", "visible_style_code")} if vis else None,
               "winner": _cand_summary(winner), "expected_candidate": _cand_summary(expected_cand),
               "timings_ms": r.evidence.get("timings_ms", {})}
        results.append(row)
        flag = "OK " if hit1 else ("~  " if hit3 else "XX ")
        print(f"[{i}/{len(cases)}] {flag} {r.status:<10} {c['expected_style_code']:<14} got {row['got'] or '-':<14} "
              f"{r.confidence if r.confidence is not None else 0:.2f} {ms}ms {','.join(r.failure_codes)}")

    n = len(cases)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    print("\n── Summary ──")
    print(f"cases                       {n}")
    print(f"top-1 accuracy              {top1 / n:.3f}")
    print(f"top-3 accuracy              {top3 / n:.3f}")
    print(f"confident (high/medium)     {confident / n:.3f}")
    print(f"SKU accuracy | confident    {exact / confident:.3f}" if confident else "SKU accuracy | confident    n/a")
    print(f"StockX mapping | confident  {mapped / confident:.3f}" if confident else "StockX mapping | confident  n/a")
    print(f"FALSE-POSITIVE rate         {confident_wrong / n:.3f}   (confident but wrong — must stay near 0)")
    print(f"unresolved rate             {unresolved / n:.3f}")
    print(f"latency mean/median (ms)    {statistics.mean(latencies):.0f} / {statistics.median(latencies):.0f}")
    print(f"results → {args.out}")


if __name__ == "__main__":
    main()
