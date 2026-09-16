"""
Run one identification from the command line, without the server:

    python scripts/identify_cli.py photo.jpg [photo2.jpg ...] [--url PRODUCT_URL] [--title "..."] [--json]

Uses the DB (reference index, cache) when reachable; degrades otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import db_available                 # noqa: E402
from app.pipeline.identify import Pipeline, RunInput     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="Photo path(s)")
    ap.add_argument("--url", default="", help="Retailer product page URL")
    ap.add_argument("--title", default="")
    ap.add_argument("--description", default="")
    ap.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = ap.parse_args()

    images = [Path(p).read_bytes() for p in args.images]
    db_ok = db_available()
    if not db_ok:
        print("(database unreachable — running without the reference index/cache)")
    pipeline = Pipeline(db_enabled=db_ok)
    result = pipeline.run_inline(RunInput(images=images, product_url=args.url, title=args.title,
                                          description=args.description),
                                 on_update=lambda ctx: _print_stage(ctx))
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return
    print(f"\nSTATUS: {result.status.upper()}  confidence {result.confidence}")
    if result.product:
        p = result.product
        print(f"  {p.brand} {p.model} — {p.colorway}\n  style code: {p.style_code}  ({p.gender}/{p.size_category})")
    if result.stockx:
        s = result.stockx
        print(f"  StockX: {s.product_name} {s.url}\n  lowest ask ${s.lowest_ask}  highest bid ${s.highest_bid}")
    if result.failure_codes:
        print(f"  flags: {', '.join(result.failure_codes)}")
    print("\nCandidates:")
    for c in result.candidates[:8]:
        mark = "x" if c.rejected else " "
        print(f" [{mark}] {c.score:.3f} {c.style_code_display:<14} {c.name[:50]:<50} {','.join(c.contradictions)[:60]}")


_printed = set()


def _print_stage(ctx):
    for st in ctx.stages:
        if st.status in ("done", "skipped", "failed") and st.name not in _printed:
            _printed.add(st.name)
            print(f"  {st.status:<8} {st.label:<28} {st.detail}")


if __name__ == "__main__":
    main()
