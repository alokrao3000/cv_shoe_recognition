"""
Re-derives products.model / sub_model / gender / size_category from each
product's name with the current parser (app/extraction/normalize.py). Run
after improving the model/gender patterns so already-seeded rows benefit.

    python scripts/reparse_products.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.db.models import Product                 # noqa: E402
from app.db.session import db_session             # noqa: E402
from app.extraction.normalize import parse_title  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    changed = 0
    with db_session() as s:
        for p in s.execute(select(Product)).scalars():
            if not p.name:
                continue
            parsed = parse_title(p.name, p.brand or "")
            updates = {}
            if parsed.model and parsed.model != (p.model or ""):
                updates["model"] = parsed.model
            if parsed.sub_model and parsed.sub_model != (p.sub_model or ""):
                updates["sub_model"] = parsed.sub_model
            if parsed.gender != "unknown" and not p.gender:
                updates["gender"] = parsed.gender
            if parsed.size_category != "unknown" and not p.size_category:
                updates["size_category"] = parsed.size_category
            if updates:
                changed += 1
                if args.dry_run:
                    print(f"{p.style_code_display or p.style_code}: {p.name[:50]!r} -> {updates}")
                else:
                    for k, v in updates.items():
                        setattr(p, k, v)
        if args.dry_run:
            s.rollback()
    print(f"{changed} product(s) {'would be' if args.dry_run else ''} updated")


if __name__ == "__main__":
    main()
