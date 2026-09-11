# CV Shoe Recognition

Given a shoe photo (typically one a scraper couldn't extract a SKU from),
identify which shoe it is via visual similarity search, then look up its
current StockX lowest ask / highest bid.

```
photo → embed (DINOv2) → nearest neighbor in reference index → SKU
      → StockX catalog match → lowest ask / highest bid
```

This is a companion to `sneaker-arbitrage` (`C:\Users\raoal\sneaker-arbitrage`),
not a replacement for it — that project's scrapers already extract a SKU
directly from most retailer pages; this project exists for the products
where that fails (`sku_parse_failed` there). It's standalone: no shared
runtime dependency, no database of its own. See **Reference data** below for
the one place it *reads* from that project's DB.

## Setup

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then fill in values
```

### StockX API

1. Create/reuse a developer app at developer.stockx.com; put
   `STOCKX_CLIENT_ID`, `STOCKX_CLIENT_SECRET`, `STOCKX_API_KEY` in `.env`.
2. Register this project's redirect URI (`STOCKX_REDIRECT_URI`, default
   `http://localhost:8018/stockx/callback` — a different port than
   sneaker-arbitrage's 8017, so both can be registered on the same app).
3. Run `python scripts/stockx_auth.py` once — interactive login, saves a
   refresh token to `data/stockx_token.json` (git-ignored). Get **this
   project its own refresh token** rather than copying sneaker-arbitrage's —
   two processes rotating the same one can race and invalidate each other's
   session. See `app/stockx_client.py`'s docstring for the full auth model.

### Reference data

The reference index (the "known shoes" the model matches photos against) is
built from `sneaker-arbitrage`'s Postgres DB: every distinct SKU its
scrapers successfully parsed already has a retailer photo attached
(`supplier_products.image_url`). That's free, real-world labeled data — no
extra scraping needed to bootstrap this project.

1. Make sure that DB is reachable (from the other repo: `docker compose up
   -d postgres`, or its `runprogram` command).
2. Set `ARBITRAGE_DATABASE_URL` in `.env` (defaults to the same local
   connection string that project uses).
3. Build the index:
   ```
   python scripts/build_reference_index.py          # full build
   python scripts/build_reference_index.py --limit 200   # quick test build
   ```
   This downloads one photo per distinct SKU and embeds it — a full build
   over thousands of SKUs takes a few minutes on CPU. Re-run periodically
   (e.g. weekly) to pick up newly-seen SKUs; it's a full rebuild each time,
   which is fine at this scale.

## Running it

```
venv\Scripts\python -m uvicorn app.main:app --reload --port 8100
```

Open http://localhost:8100 for a minimal upload page, or `POST` an image to
`/identify` (multipart `file` field) directly. `/docs` has the full API.

For quick manual testing without the server:
```
python scripts/identify_cli.py path\to\photo.jpg
```

## How matching works

- **Embedding model**: DINOv2 (`facebook/dinov2-base`, configurable via
  `EMBEDDING_MODEL`) — chosen over CLIP because this is fine-grained
  *instance* retrieval (telling colorways of the same silhouette apart), not
  loose semantic matching. See `app/embeddings.py`.
- **Index**: L2-normalized embeddings in a NumPy array, searched by a plain
  matrix-multiply (dot product = cosine similarity). Not FAISS — that was
  tried first and dropped after it turned out to crash on import alongside
  torch on Windows (`OMP: Error #15`, conflicting bundled OpenMP runtimes;
  reproducible). A brute-force NumPy scan is exact and just as fast at this
  scale (thousands of SKUs) and has no such conflict. See `app/index.py`.
- **Confidence gate**: a match is only trusted when the top candidate's
  similarity clears `MIN_MATCH_SIMILARITY` *and* beats the runner-up by
  `MIN_MATCH_MARGIN`. Otherwise the response is `identified: false` with a
  `reason` (`no_confident_match` / `ambiguous_top_match`) — returning nothing
  is better than a wrong SKU silently feeding a price lookup. Tune both in
  `.env` once you've seen real match-score distributions on your data.

## StockX lookup

`app/stockx_client.py` is a trimmed port of sneaker-arbitrage's
`app/scrapers/stockx_api.py` — same auth model and catalog-match strategy
(style-code equality, then name+colorway fuzzy fallback), same market-data
parsing. Differences, both because this project's call volume is on-demand
and low (a couple of API calls per identify request, not a bulk scrape):
- Refresh-token persistence is a local JSON file, not a DB table.
- The daily rate-limit counter is in-memory (resets on restart), not shared
  across processes. **If you run this alongside sneaker-arbitrage at real
  scrape volume against the same StockX account, be aware the two
  self-throttle independently and don't share a live budget counter** — fine
  for occasional identify calls, not a substitute for the DB-backed counter
  sneaker-arbitrage uses for its bulk lookups.

`/identify` returns the overall lowest ask / highest bid (min/max across all
available sizes) plus a per-size breakdown.

## Known limitations / next steps

- One reference image per SKU (whichever retailer photo was most recently
  scraped). Multiple angles per SKU would likely improve match robustness —
  `supplier_products` only stores one `image_url` today, so this would need
  a schema change on the sneaker-arbitrage side first.
- No fine-tuning — DINOv2 is used zero-shot. If match quality on real data
  isn't good enough, the next step is fine-tuning on scraped (sku, image)
  pairs rather than swapping the base model.
- Not yet wired into sneaker-arbitrage's `sku_parse_failed` fallback path —
  by design, per the standalone-first decision. Integration would mean that
  scraper calling this project's `/identify` (or importing its functions
  directly) when SKU extraction fails.
