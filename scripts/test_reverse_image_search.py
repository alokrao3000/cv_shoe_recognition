"""
Manual test/diagnostic for app/reverse_image_search.py — run this directly
against a real photo to see whether Google's reverse-image-search upload
gets through from your network, or hits the same bot-check wall it hit in
the original dev environment (see that module's docstring).

Usage:
    python scripts/test_reverse_image_search.py path\\to\\photo.jpg [--headed]

--headed opens a real browser window so you can see directly whether it's a
"I'm not a robot" challenge page or real results — same idea as
scripts/backfill_stockx_images.py --headed for the StockX side.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import reverse_image_search                     # noqa: E402
from app.browser_session import BrowserSession            # noqa: E402
from app.config import settings                            # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--headed", action="store_true",
                        help="Show the browser window — useful to see directly "
                             "whether it's a CAPTCHA/challenge page or real results.")
    args = parser.parse_args()

    data = Path(args.image_path).read_bytes()

    with BrowserSession(platform="google_images", headless=not args.headed,
                         proxy_url=settings.reverse_image_search_proxy_url,
                         state_dir=settings.browser_state_dir,
                         nav_timeout_ms=settings.browser_nav_timeout_ms) as session:
        outcome = reverse_image_search.search_by_image(session, data, Path(args.image_path).name)

    if outcome.blocked:
        print(f"BLOCKED: {outcome.blocked_reason}")
        print("Not past the bot check on this network — see app/reverse_image_search.py's "
              "docstring for the proxy option.")
        return

    print(f"best_guess: {outcome.best_guess!r}")
    print(f"result links ({len(outcome.result_links)}):")
    for link in outcome.result_links:
        print(f"  {link.url}  —  {link.title!r}")

    # Same extraction find_sku() applies, shown inline here rather than
    # calling find_sku() itself (that would launch a second browser session).
    sku = reverse_image_search.extract_sku_from_text(outcome.best_guess or "")
    source = "best_guess"
    if not sku:
        for link in outcome.result_links:
            sku = reverse_image_search.extract_sku_from_text(link.title)
            if sku:
                source = link.url
                break
    print(f"\nSKU from best_guess/titles -> sku={sku!r}, source={source if sku else None!r}")
    if not sku:
        print("(find_sku() would go on to fetch each result page's full text next — "
              "not repeated here to avoid a second browser session.)")


if __name__ == "__main__":
    main()
