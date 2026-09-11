"""
Quick manual test without running the server:

    python scripts/identify_cli.py path/to/photo.jpg [--no-market]

Prints the top candidates and, for a confident match, the StockX lowest
ask / highest bid.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embeddings, stockx_client                  # noqa: E402
from app.index import ReferenceIndex, classify_match        # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a shoe photo.")
    parser.add_argument("--no-market", action="store_true", help="Skip the StockX lookup.")
    args = parser.parse_args()

    index = ReferenceIndex.load()
    print(f"Loaded reference index: {len(index)} SKUs.")

    image = embeddings.load_image(args.image)
    vec = embeddings.embed_image(image)
    candidates = index.search(vec)

    print("\nTop candidates:")
    for c in candidates:
        print(f"  {c.similarity:.4f}  {c.sku:<20} {c.name or ''}")

    identified, reason = classify_match(candidates)
    if not identified:
        print(f"\nNot identified ({reason}).")
        return

    top = candidates[0]
    print(f"\nIdentified: {top.sku} ({top.name or 'unknown name'})")

    if args.no_market:
        return
    if not stockx_client.is_configured():
        print("StockX credentials not configured (.env) — skipping market lookup.")
        return

    client = stockx_client.StockXAPIClient()
    try:
        market, err = client.get_market(top.sku, top.name or "")
    finally:
        client.close()

    if market is None:
        print(f"StockX lookup failed: {err}")
        return
    print(f"StockX: {market.url}")
    print(f"  Lowest ask:  {market.lowest_ask}")
    print(f"  Highest bid: {market.highest_bid}")
    for s in market.sizes:
        print(f"    size {s.size:<6} ask={s.lowest_ask}  bid={s.highest_bid}")


if __name__ == "__main__":
    main()
